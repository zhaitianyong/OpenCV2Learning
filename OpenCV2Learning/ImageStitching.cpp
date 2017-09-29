/************************************************************************
> File Name: 
> Author:atway
> Mail:atway#126.com(#=>@)
> Created Time: 2014年10月15日 星期三 12时00分33秒
************************************************************************/

#include "stdafx.h"
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>  
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\nonfree\features2d.hpp>
#include <opencv2\legacy\legacy.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <iostream>  
#include <math.h>

using namespace std;
using namespace cv;


Mat imageStitching(Mat& targetImage,Mat& srcImage)
{
	cout << "正在灰度图转换" << endl;
	//灰度图转换  
	Mat image1, image2;
	/*cvtColor(srcImage, image1, CV_RGB2GRAY);
	cvtColor(targetImage, image2, CV_RGB2GRAY);*/
	image1 = srcImage.clone();
	image2 = targetImage.clone();
	cout << "正在提取特征点" << endl;
	//提取特征点    
	SiftFeatureDetector siftDetector;  //hessian阈值
	vector<KeyPoint> keyPoint1, keyPoint2;
	siftDetector.detect(image1, keyPoint1);
	siftDetector.detect(image2, keyPoint2);

	//Mat dstImg1, dstImg2;
	//drawKeypoints(srcImage, keyPoint1, dstImg1);
	//drawKeypoints(targetImage, keyPoint2, dstImg2);
	//imshow("dstImg1", dstImg1);
	//imshow("dstImg2", dstImg2);
	//waitKey(0);
	//特征点描述，为下边的特征点匹配做准备    
	cout << "正在特征点描述" << endl;
	SiftDescriptorExtractor SiftDescriptor;
	Mat imageDesc1, imageDesc2;
	SiftDescriptor.compute(image1, keyPoint1, imageDesc1);
	SiftDescriptor.compute(image2, keyPoint2, imageDesc2);

	cout << "正在匹配特征点" << endl;
	//获得匹配特征点，并提取最优配对   
	FlannBasedMatcher matcher;
	vector<DMatch> matchPoints;
	matcher.match(imageDesc1, imageDesc2, matchPoints, Mat());

	cout << "获取排在前N个的最优匹配特征点 " << endl;
	sort(matchPoints.begin(), matchPoints.end()); //特征点排序 
	float minDistance = matchPoints[0].distance;
	vector<Point2f> imagePoints1, imagePoints2; //获取排在前N个的最优匹配特征点 
	vector<DMatch> goodMatchPoints;
	cout << "total count:" << matchPoints.size() << endl;
	for (int i = 0; i<matchPoints.size(); i++)
	{
		if (matchPoints[i].distance>2 * minDistance) break;
		goodMatchPoints.push_back(matchPoints[i]);
		imagePoints1.push_back(keyPoint1[matchPoints[i].queryIdx].pt);
		imagePoints2.push_back(keyPoint2[matchPoints[i].trainIdx].pt);
	}
	cout << "match count:" << imagePoints1.size() << endl;

	Mat imgMatches;
	drawMatches(srcImage, keyPoint1, targetImage, keyPoint2, goodMatchPoints, imgMatches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	namedWindow("匹配", WINDOW_NORMAL);
	imshow("匹配", imgMatches);
	waitKey(0);
	//计算旋转矩阵
	//获取左边图像到右边图像的投影映射关系  
	cout << "计算旋转矩阵H " << endl;
	Mat H = findHomography(imagePoints1, imagePoints2, CV_RANSAC);
	cout << "正在图像拼接 " << endl;
	Mat stitchedImage;
	warpPerspective(srcImage, stitchedImage, H, targetImage.size());
	namedWindow("匹配", WINDOW_NORMAL);
	imshow("匹配", stitchedImage);
	waitKey(0);
	//重叠区域 进行灰度拉伸
	/*Mat half(stitchedImage, Rect(0, 0, targetImage.cols, targetImage.rows));
	targetImage.copyTo(half);*/
	addWeighted(stitchedImage, 0.5, targetImage, 0.5, 0, stitchedImage);
	imwrite("data\\Micro4\\tmp.jpg", stitchedImage);
	return stitchedImage;
}
/*
//提取特征点和特征点描述（128维特征向量）
*/
void getkeyPoints(Mat& img, vector<KeyPoint>& keyPoint, Mat& imageDesc)
{
	SiftFeatureDetector siftDetector;// siftDetector(0, 3, 0.04); //默认0.04
	SiftDescriptorExtractor SiftDescriptor;
	cout << "正在提取特征点" << endl;
	siftDetector.detect(img, keyPoint);
	cout << "正在特征点描述" << endl;
	SiftDescriptor.compute(img, keyPoint, imageDesc);
}

/*
//图片融合 计算平均值
*/
void blendImage(Mat& input, Mat& target, Mat& output, Mat& H ,bool dir=0)
{
	assert(input.channels() == 3);
	assert(target.channels() == 3);
	output = target.clone();
	for (size_t row = 1; row < input.rows-1; row++)
	{
		for (size_t col = 1; col < input.cols-1; col++)
		{
			bool flag = false;
			for (size_t i = 0; i < 3; i++)
			{
				if (input.at<Vec3b>(row, col)[i] != 0)
				{
					flag = true;
					break;
				}
			}
			//判断是否为黑色 
			//如果不为黑色就取平均值
			if (flag)
			{
				for (size_t i = 0; i < 3; i++)
				{
					//
					output.at<Vec3b>(row, col)[i] = (output.at<Vec3b>(row, col)[i] + input.at<Vec3b>(row, col)[i]) / 2;
				}
			}
		}
	}
}

void refineMatcheswithHomography(vector<DMatch>& matches, vector<KeyPoint>& tmpKeyPoint, vector<KeyPoint>& targetKeyPoint, double reprojectionThreshold, Mat& homography) {
	const int minNumbermatchesAllowed = 4;
	if (matches.size() < minNumbermatchesAllowed)
		return;
	//Prepare data for findHomography
	vector<Point2f> srcPoints(matches.size());
	vector<Point2f> dstPoints(matches.size());

	for (size_t i = 0; i < matches.size(); i++) {
		srcPoints[i] = targetKeyPoint[matches[i].trainIdx].pt;
		dstPoints[i] = tmpKeyPoint[matches[i].queryIdx].pt;
	}

	//find homography matrix and get inliers mask
	vector<uchar> inliersMask(srcPoints.size());
	homography = findHomography(srcPoints, dstPoints, CV_FM_RANSAC, reprojectionThreshold, inliersMask);

	vector<DMatch> inliers;
	for (size_t i = 0; i < inliersMask.size(); i++) {
		if (inliersMask[i])
			inliers.push_back(matches[i]);
	}
	matches.swap(inliers);
}

void knnMatch(BFMatcher& matcher, Mat& queryDescriptors, Mat& trainDescriptors, vector<DMatch>& matches) {

	const float minRatio = 1.f / 1.5f;
	const int k = 2;

	vector<vector<DMatch>> knnMatches;
	matcher.knnMatch(queryDescriptors, trainDescriptors, knnMatches, k);

	for (size_t i = 0; i < knnMatches.size(); i++) {
		const DMatch& bestMatch = knnMatches[i][0];
		const DMatch& betterMatch = knnMatches[i][1];

		float  distanceRatio = bestMatch.distance / betterMatch.distance;
		if (distanceRatio < minRatio)
			matches.push_back(bestMatch);
	}
}

//特征拼接
void main_feater_stitching()
{
	Mat targetImage = imread("data\\Micro4\\Sample0_0.jpg");
	if (targetImage.empty())
	{
		cout << "load image failed" << endl;
		return;
	}
	//提取目标图片的SIFT特征
	vector<KeyPoint> keyPoint;
	Mat imageDesc;
	getkeyPoints(targetImage, keyPoint, imageDesc);
	//提取其它待处理的图像
	vector<string> files = {"data\\Micro4\\Sample0_1.jpg","data\\Micro4\\Sample0_2.jpg","data\\Micro4\\Sample0_3.jpg","data\\Micro4\\Sample0_4.jpg" };
	vector<Mat> blendImages;
	for (size_t i = 0; i < files.size(); i++)
	{
		vector<KeyPoint> tmpKeyPoint;
		Mat tmpImageDesc;
		Mat srcImage = imread(files[i]);
		getkeyPoints(srcImage, tmpKeyPoint, tmpImageDesc);
		cout << "正在匹配特征点" << endl;
		//获得匹配特征点，并提取最优配对   
		//FlannBasedMatcher matcher;
		//http://www.cnblogs.com/wangguchangqing/p/4333873.html
		BFMatcher matcher;// (NORM_L2, true); //暴力匹配，反向去除匹配的点
		vector<DMatch> matches;
		Mat H;
		knnMatch(matcher, tmpImageDesc, imageDesc, matches); //KNN 选择匹配点对
		//matcher.match(tmpImageDesc, imageDesc, matchPoints, Mat());
		cout << "原始 total count:" << matches.size() << endl;
		//进一步赛选匹配点对
		refineMatcheswithHomography(matches, tmpKeyPoint, keyPoint, 3, H);
		cout << "单应矩阵赛选后 total count:" << matches.size() << endl;
		//获得匹配点数
		cout << "获取排在前N个的最优匹配特征点 " << endl;
		sort(matches.begin(), matches.end()); //特征点排序 
		float minDistance = matches[0].distance;
		vector<Point2f> targetImagePoints, imagePoints; //获取排在前N个的最优匹配特征点 
		vector<DMatch> goodMatchPoints;
		cout << "total count:" << matches.size() << endl;
		for (int j = 0; j<matches.size(); j++)
		{
			//if (matchPoints[j].distance>2 * minDistance) break;
			goodMatchPoints.push_back(matches[j]);
			imagePoints.push_back(tmpKeyPoint[matches[j].queryIdx].pt);
			targetImagePoints.push_back(keyPoint[matches[j].trainIdx].pt);
		}
		cout << "match count:" << targetImagePoints.size() << endl;
		//
	   /*Mat imgMatches;
		drawMatches(srcImage, tmpKeyPoint, targetImage, keyPoint, goodMatchPoints, imgMatches, Scalar::all(-1), Scalar::all(-1),
			vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		namedWindow("匹配", WINDOW_NORMAL);
		imshow("匹配", imgMatches);
		waitKey(0);*/
		//H矩阵
		cout << "计算旋转矩阵H " << endl;
	    H = findHomography(imagePoints, targetImagePoints, CV_RANSAC);
		cout << "正在图像拼接 " << endl;
		Mat stitchedImage, outImage;
		warpPerspective(srcImage, stitchedImage, H, Size(targetImage.cols,targetImage.rows));
		//图像融合
		blendImage(stitchedImage,targetImage, outImage,H,0); //0 右，1 左
		blendImages.push_back(outImage);
		imwrite("data\\Micro4\\tmp.jpg", outImage);
	}

	//计算平均值
	//Mat dstImage;
	//dstImage = Mat::zeros(targetImage.size(),targetImage.type());
	Mat tmpImage1, tmpImage2, dstImage;

	addWeighted(blendImages[0], 0.5, blendImages[1], 0.5, 0, tmpImage1);

	addWeighted(blendImages[2], 0.5, blendImages[3], 0.5, 0, tmpImage2);

	addWeighted(tmpImage1, 0.5, tmpImage2, 0.5, 0, dstImage);
	
	//匹配
	imwrite("data\\Micro4\\Sample0.jpg", dstImage);

}

//图像拼接
/*
void main_stich01(int argc, char** argv)
{

	Mat targetImage = imread("data\\Micro4\\Sample0_0.jpg");
	if (targetImage.empty())
	{
		cout << "load image failed" << endl;
		return;
	}
	cout << "图片拼接开始" << endl;
	vector<string> files = {"data\\Micro4\\Sample0_1.jpg","data\\Micro4\\Sample0_2.jpg","data\\Micro4\\Sample0_3.jpg","data\\Micro4\\Sample0_4.jpg"};
	for (size_t i = 0; i <files.size(); i++)
	{
		cout << "正在拼接第" << i+1 << "张图片...." << endl;
		Mat img = imread(files[i]);
		Mat stitchedImage =imageStitching(targetImage, img);
		//targetImage = stitchedImage;
		cout << "第" << i + 1 << "张图片拼接结束" << endl;
	}
	cout << "图片拼接结束" << endl;
	imwrite("data\\Micro4\\result.jpg", targetImage);
}
*/



// hard stitching
void main_hard_stitching()
{
	Mat srcImage, dstImage;
    srcImage = imread("data\\Micro4\\Sample0_0.jpg");
	//提取其它待处理的图像
	vector<string> files = { "data\\Micro4\\Sample0_1.jpg","data\\Micro4\\Sample0_2.jpg","data\\Micro4\\Sample0_3.jpg","data\\Micro4\\Sample0_4.jpg" };
	Mat rightTop = imread(files[0]);
	Mat rightBottom = imread(files[1]);
	Mat leftBottom = imread(files[2]);
	Mat leftTop = imread(files[3]);

	int height = srcImage.size().height;
	int width = srcImage.size().width;

 	int h = height / 2;
	int w = width / 2;

    //提取ROI
	Rect rtRect, rbRect, ltRect, lbRect;
	rtRect = Rect(width/4, height/4, w, h);
	rbRect = Rect(width / 4-20, height / 4+20, w, h);
	ltRect = Rect(width / 4, height / 4, w, h);
	lbRect = Rect(width / 4, height / 4, w, h);
	Mat rtROI = rightTop(rtRect);
	Mat rbROI = rightBottom(rbRect);
	Mat lbROI = leftBottom(lbRect);
	Mat ltROI = leftTop(ltRect);


	namedWindow("image", WINDOW_NORMAL);
	/*rectangle(rightTop, rtRect, Scalar(0, 0, 255), 2);
	imshow("image",rightTop);
	waitKey(0);
	rectangle(rightBottom, rbRect, Scalar(0, 0, 255), 2);
	imshow("image", rightBottom);
	waitKey(0);
	rectangle(leftBottom, lbRect, Scalar(0, 0, 255), 2);
	imshow("image", leftBottom);
	waitKey(0);
	rectangle(leftTop, ltRect, Scalar(0, 0, 255), 2);
	imshow("image", leftTop);
	waitKey(0);*/

	//简单求平均值
	dstImage = Mat::zeros(srcImage.size(),srcImage.type());

	//拆分成9个区域进行计算平均值
	/*for (size_t row = 0; row < dstImage.rows; row++)
	{
		for (size_t col = 0; col < dstImage.cols ; col++)
		{
			for (size_t i = 0; i < dstImage.channels(); i++)
			{
				dstImage.at<Vec3b>(row, col)[i]=
			}
		}
	}*/
	//把所有的ROI映射到同一个平面上
	Mat imageROI;
    imageROI = dstImage(Rect(width/2,0,w,h));
	rtROI.copyTo(imageROI);
	imshow("image", dstImage);
	waitKey(0);
    imageROI = dstImage(Rect(width/2, height/2, w, h));
	rbROI.copyTo(imageROI);
	imshow("image", dstImage);
	waitKey(0);
	imageROI = dstImage(Rect(0, height/2, w, h));
	lbROI.copyTo(imageROI);
	imshow("image", dstImage);
	waitKey(0);
	imageROI = dstImage(Rect(0, 0, w, h));
	ltROI.copyTo(imageROI);
	imshow("image", dstImage);
	waitKey(0);

	addWeighted(srcImage, 0.5, dstImage, 0.5, 0, dstImage);
	imwrite("data\\Micro4\\result.jpg", dstImage);
}
/************************************************************************
> File Name: 
> Author:atway
> Mail:atway#126.com(#=>@)
> Created Time: 2014��10��15�� ������ 12ʱ00��33��
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
	cout << "���ڻҶ�ͼת��" << endl;
	//�Ҷ�ͼת��  
	Mat image1, image2;
	/*cvtColor(srcImage, image1, CV_RGB2GRAY);
	cvtColor(targetImage, image2, CV_RGB2GRAY);*/
	image1 = srcImage.clone();
	image2 = targetImage.clone();
	cout << "������ȡ������" << endl;
	//��ȡ������    
	SiftFeatureDetector siftDetector;  //hessian��ֵ
	vector<KeyPoint> keyPoint1, keyPoint2;
	siftDetector.detect(image1, keyPoint1);
	siftDetector.detect(image2, keyPoint2);

	//Mat dstImg1, dstImg2;
	//drawKeypoints(srcImage, keyPoint1, dstImg1);
	//drawKeypoints(targetImage, keyPoint2, dstImg2);
	//imshow("dstImg1", dstImg1);
	//imshow("dstImg2", dstImg2);
	//waitKey(0);
	//������������Ϊ�±ߵ�������ƥ����׼��    
	cout << "��������������" << endl;
	SiftDescriptorExtractor SiftDescriptor;
	Mat imageDesc1, imageDesc2;
	SiftDescriptor.compute(image1, keyPoint1, imageDesc1);
	SiftDescriptor.compute(image2, keyPoint2, imageDesc2);

	cout << "����ƥ��������" << endl;
	//���ƥ�������㣬����ȡ�������   
	FlannBasedMatcher matcher;
	vector<DMatch> matchPoints;
	matcher.match(imageDesc1, imageDesc2, matchPoints, Mat());

	cout << "��ȡ����ǰN��������ƥ�������� " << endl;
	sort(matchPoints.begin(), matchPoints.end()); //���������� 
	float minDistance = matchPoints[0].distance;
	vector<Point2f> imagePoints1, imagePoints2; //��ȡ����ǰN��������ƥ�������� 
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
	namedWindow("ƥ��", WINDOW_NORMAL);
	imshow("ƥ��", imgMatches);
	waitKey(0);
	//������ת����
	//��ȡ���ͼ���ұ�ͼ���ͶӰӳ���ϵ  
	cout << "������ת����H " << endl;
	Mat H = findHomography(imagePoints1, imagePoints2, CV_RANSAC);
	cout << "����ͼ��ƴ�� " << endl;
	Mat stitchedImage;
	warpPerspective(srcImage, stitchedImage, H, targetImage.size());
	namedWindow("ƥ��", WINDOW_NORMAL);
	imshow("ƥ��", stitchedImage);
	waitKey(0);
	//�ص����� ���лҶ�����
	/*Mat half(stitchedImage, Rect(0, 0, targetImage.cols, targetImage.rows));
	targetImage.copyTo(half);*/
	addWeighted(stitchedImage, 0.5, targetImage, 0.5, 0, stitchedImage);
	imwrite("data\\Micro4\\tmp.jpg", stitchedImage);
	return stitchedImage;
}
/*
//��ȡ�������������������128ά����������
*/
void getkeyPoints(Mat& img, vector<KeyPoint>& keyPoint, Mat& imageDesc)
{
	SiftFeatureDetector siftDetector;// siftDetector(0, 3, 0.04); //Ĭ��0.04
	SiftDescriptorExtractor SiftDescriptor;
	cout << "������ȡ������" << endl;
	siftDetector.detect(img, keyPoint);
	cout << "��������������" << endl;
	SiftDescriptor.compute(img, keyPoint, imageDesc);
}

/*
//ͼƬ�ں� ����ƽ��ֵ
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
			//�ж��Ƿ�Ϊ��ɫ 
			//�����Ϊ��ɫ��ȡƽ��ֵ
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

//����ƴ��
void main_feater_stitching()
{
	Mat targetImage = imread("data\\Micro4\\Sample0_0.jpg");
	if (targetImage.empty())
	{
		cout << "load image failed" << endl;
		return;
	}
	//��ȡĿ��ͼƬ��SIFT����
	vector<KeyPoint> keyPoint;
	Mat imageDesc;
	getkeyPoints(targetImage, keyPoint, imageDesc);
	//��ȡ�����������ͼ��
	vector<string> files = {"data\\Micro4\\Sample0_1.jpg","data\\Micro4\\Sample0_2.jpg","data\\Micro4\\Sample0_3.jpg","data\\Micro4\\Sample0_4.jpg" };
	vector<Mat> blendImages;
	for (size_t i = 0; i < files.size(); i++)
	{
		vector<KeyPoint> tmpKeyPoint;
		Mat tmpImageDesc;
		Mat srcImage = imread(files[i]);
		getkeyPoints(srcImage, tmpKeyPoint, tmpImageDesc);
		cout << "����ƥ��������" << endl;
		//���ƥ�������㣬����ȡ�������   
		//FlannBasedMatcher matcher;
		//http://www.cnblogs.com/wangguchangqing/p/4333873.html
		BFMatcher matcher;// (NORM_L2, true); //����ƥ�䣬����ȥ��ƥ��ĵ�
		vector<DMatch> matches;
		Mat H;
		knnMatch(matcher, tmpImageDesc, imageDesc, matches); //KNN ѡ��ƥ����
		//matcher.match(tmpImageDesc, imageDesc, matchPoints, Mat());
		cout << "ԭʼ total count:" << matches.size() << endl;
		//��һ����ѡƥ����
		refineMatcheswithHomography(matches, tmpKeyPoint, keyPoint, 3, H);
		cout << "��Ӧ������ѡ�� total count:" << matches.size() << endl;
		//���ƥ�����
		cout << "��ȡ����ǰN��������ƥ�������� " << endl;
		sort(matches.begin(), matches.end()); //���������� 
		float minDistance = matches[0].distance;
		vector<Point2f> targetImagePoints, imagePoints; //��ȡ����ǰN��������ƥ�������� 
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
		namedWindow("ƥ��", WINDOW_NORMAL);
		imshow("ƥ��", imgMatches);
		waitKey(0);*/
		//H����
		cout << "������ת����H " << endl;
	    H = findHomography(imagePoints, targetImagePoints, CV_RANSAC);
		cout << "����ͼ��ƴ�� " << endl;
		Mat stitchedImage, outImage;
		warpPerspective(srcImage, stitchedImage, H, Size(targetImage.cols,targetImage.rows));
		//ͼ���ں�
		blendImage(stitchedImage,targetImage, outImage,H,0); //0 �ң�1 ��
		blendImages.push_back(outImage);
		imwrite("data\\Micro4\\tmp.jpg", outImage);
	}

	//����ƽ��ֵ
	//Mat dstImage;
	//dstImage = Mat::zeros(targetImage.size(),targetImage.type());
	Mat tmpImage1, tmpImage2, dstImage;

	addWeighted(blendImages[0], 0.5, blendImages[1], 0.5, 0, tmpImage1);

	addWeighted(blendImages[2], 0.5, blendImages[3], 0.5, 0, tmpImage2);

	addWeighted(tmpImage1, 0.5, tmpImage2, 0.5, 0, dstImage);
	
	//ƥ��
	imwrite("data\\Micro4\\Sample0.jpg", dstImage);

}

//ͼ��ƴ��
/*
void main_stich01(int argc, char** argv)
{

	Mat targetImage = imread("data\\Micro4\\Sample0_0.jpg");
	if (targetImage.empty())
	{
		cout << "load image failed" << endl;
		return;
	}
	cout << "ͼƬƴ�ӿ�ʼ" << endl;
	vector<string> files = {"data\\Micro4\\Sample0_1.jpg","data\\Micro4\\Sample0_2.jpg","data\\Micro4\\Sample0_3.jpg","data\\Micro4\\Sample0_4.jpg"};
	for (size_t i = 0; i <files.size(); i++)
	{
		cout << "����ƴ�ӵ�" << i+1 << "��ͼƬ...." << endl;
		Mat img = imread(files[i]);
		Mat stitchedImage =imageStitching(targetImage, img);
		//targetImage = stitchedImage;
		cout << "��" << i + 1 << "��ͼƬƴ�ӽ���" << endl;
	}
	cout << "ͼƬƴ�ӽ���" << endl;
	imwrite("data\\Micro4\\result.jpg", targetImage);
}
*/



// hard stitching
void main_hard_stitching()
{
	Mat srcImage, dstImage;
    srcImage = imread("data\\Micro4\\Sample0_0.jpg");
	//��ȡ�����������ͼ��
	vector<string> files = { "data\\Micro4\\Sample0_1.jpg","data\\Micro4\\Sample0_2.jpg","data\\Micro4\\Sample0_3.jpg","data\\Micro4\\Sample0_4.jpg" };
	Mat rightTop = imread(files[0]);
	Mat rightBottom = imread(files[1]);
	Mat leftBottom = imread(files[2]);
	Mat leftTop = imread(files[3]);

	int height = srcImage.size().height;
	int width = srcImage.size().width;

 	int h = height / 2;
	int w = width / 2;

    //��ȡROI
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

	//����ƽ��ֵ
	dstImage = Mat::zeros(srcImage.size(),srcImage.type());

	//��ֳ�9��������м���ƽ��ֵ
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
	//�����е�ROIӳ�䵽ͬһ��ƽ����
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
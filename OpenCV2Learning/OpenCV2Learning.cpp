// OpenCV2Learning.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>  
#include <opencv2\nonfree\nonfree.hpp>  
#include <opencv2\legacy\legacy.hpp>
#include <iostream>  
//#include <deque>
extern "C" {
#include "vl/generic.h"
#include "vl/slic.h"
}
using namespace cv;
using namespace std;


//根据距离赛选点
//void filter_keypoint(cv::Mat &image, std::vector<cv::KeyPoint> keypoints, float minDistance, int maxCorners, float response, std::vector<cv::Point2f> &corners);

//1.sift 特征提取，并保存成文件
Point2f transform(Mat& H0,Point2f &pt)
{
	float x = pt.x;
	float y = pt.y;
	
	//Mat H=Mat::zeros(3,3,CV_32F);
	//Mat H = (Mat_<double>(3, 3) << 2.9443956105083668e-001, -7.8160232880695080e-001, 2.3137539386647455e+002, 1.5940061888353310e-001, 2.0228041496148055e-001, 3.5414308656653418e+001, -4.6702763864289168e-004, - 1.3648175637391445e-003, 1.);
	//FileStorage fs;
	//fs.open("D:\\11.xml",FileStorage::WRITE);
	//fs << "HH" << H0;
	//fs.release();
	//fs.open("D:\\11.xml", FileStorage::READ);
	//fs["HH"] >> H;
	//fs.release();
	
	float h[9] = { 2.9443956105083668e-001, -7.8160232880695080e-001, 2.3137539386647455e+002, 1.5940061888353310e-001, 2.0228041496148055e-001, 3.5414308656653418e+001, -4.6702763864289168e-004, -1.3648175637391445e-003, 1. };
	/*float h[9];
	h[0] = H.at<float>(0, 0);
	h[1] = H.at<float>(0, 1);
	h[2] = H.at<float>(0, 2);
	h[3] = H.at<float>(1, 0);
	h[4] = H.at<float>(1, 1);
	h[5] = H.at<float>(1, 2);
	h[6] = H.at<float>(2, 0);
	h[7] = H.at<float>(2, 1);
	h[8] = H.at<float>(2, 2);*/
	float newx = h[0] * x + h[1] * y + h[2];
	float newy = h[3] * x + h[4] * y + h[5];
	float iden = 1.f / (h[6] * x + h[7] * y + h[8]);

	return Point2f(newx * iden, newy * iden);
}

static char* getNextArg(int argc,char* argv[])
{
	static int idx = 1;
	if (idx < argc)	return argv[idx++];
	else	return 0;
}

Mat getDisplayImage(const Mat& img)
{
	Mat thumbnail;
	float ratio = 1.;
	if (img.size().width>1024 || img.size().height>1024)
	{
		if (img.size().width>img.size().height)
		{
			ratio = 1024. / img.size().width;
		}
		else
		{
			ratio = 1024. / img.size().height;
		}
	}
	if (ratio<1)
	{
		int w = cvRound(img.size().width*ratio);
		int h = cvRound(img.size().height*ratio);
		Size size = Size(w, h);
		thumbnail = Mat(size, img.type());
		resize(img, thumbnail, size);
		return thumbnail;
	}
	
	return img;
	
}


//生成H矩阵
int  geterateHomography(int argc, char* argv[])
{
	char* arg = getNextArg(argc, argv);
	if (!arg) return -1;
	string srcImageFile, dstImageFile, marchPointFile;
	while (arg) {
		if (!strcmp(arg, "-s")) {
			arg = getNextArg(argc, argv);
			if (!arg) return -1;

			srcImageFile = string(arg);
		}

		if (!strcmp(arg, "-d")) {
			arg = getNextArg(argc, argv);
			if (!arg) return -1;

			dstImageFile = string(arg);
		}

		if (!strcmp(arg, "-m")) {
			arg = getNextArg(argc, argv);
			if (!arg) return -1;

			marchPointFile = string(arg);
		}

		arg = getNextArg(argc, argv);
	}
	vector<Point2f> basePoints, targetPoints;
	FileStorage fs;
	fs.open(marchPointFile, FileStorage::READ);
	fs["BasePoints"] >> basePoints;
	fs["TargetPoints"] >> targetPoints;


	for (size_t i = 0; i < basePoints.size(); i++)
	{
		printf("(%f,%f),(%f,%f)\n", basePoints[i].x, basePoints[i].y, targetPoints[i].x, targetPoints[i].y);
	}

	//获取图像1到图像2的投影映射矩阵 尺寸为3*3  
	Mat H = findHomography(targetPoints, basePoints, CV_RANSAC);
	Mat srcImage, dstImage;
	srcImage = imread(srcImageFile);
	dstImage = imread(dstImageFile);

	fs.open("Homography.yaml", FileStorage::WRITE);
	fs << "H" << H;
	fs << "Size" << srcImage.size();

	Mat imageTransform;
	warpPerspective(dstImage, imageTransform, H, srcImage.size());
	imwrite("result.jpg", imageTransform);

	Mat dis = getDisplayImage(imageTransform);

	namedWindow("透视变换");
	imshow("透视变换", dis);

	waitKey(0);
	return 0;
}

//生成正视图
int  convertImageByHomography(int argc, char* argv[])
{
	char* arg = getNextArg(argc, argv);
	if (!arg) return -1;
	string srcImageFile, dstImageFile, homographyFile;
	while (arg) {
		if (!strcmp(arg, "-s")) {
			arg = getNextArg(argc, argv);
			if (!arg) return -1;

			srcImageFile = string(arg);
		}

		if (!strcmp(arg, "-d")) {
			arg = getNextArg(argc, argv);
			if (!arg) return -1;

			dstImageFile = string(arg);
		}

		if (!strcmp(arg, "-h")) {
			arg = getNextArg(argc, argv);
			if (!arg) return -1;
			homographyFile = string(arg);
		}

		arg = getNextArg(argc, argv);
	}


	FileStorage fs;
	Mat H; //= Mat::zeros(3, 3, CV_32F);
	Size size(100,100);
	fs.open(homographyFile, FileStorage::READ);
	fs["H"] >> H;
	fs["Size"] >> size;
	//fs.release();

	Mat srcImage, dstImage;
	srcImage = imread(srcImageFile);

	warpPerspective(srcImage, dstImage, H, size);

	imwrite(dstImageFile, dstImage);

	Mat dis = getDisplayImage(dstImage);

	namedWindow("透视变换");
	imshow("透视变换", dis);

	waitKey(0);


	return 0;
}



//透视投影变换，至少4组点 最后求出3*3的H矩阵
int main7(int argc,char* argv[])
{
	char *arg = getNextArg(argc,argv);
	if (!arg) return -1;
	if (!strcmp(arg, "1"))
	{
		//1 - s data\test\data\bg.jpg - d data\test\data\1.jpg - m data\test\data\match.yaml
		geterateHomography(argc, argv);
	}
	else if (!strcmp(arg, "2"))
	{
		//2 - s data\test\data\2.jpg - d data\test\data\result_2.jpg - h data\test\data\Homography.yaml
		convertImageByHomography(argc, argv);
	}
	
	return 0;
}



int main6()
{
	const string basefile = "data\\1.jpg";
	Mat srcImage, meanImage,gaussianImage;
	srcImage= imread(basefile);
	blur(srcImage, meanImage,Size(5,5));
	GaussianBlur(srcImage, gaussianImage, Size(5, 5),1.,1);
	//namedWindow("原图");
	imshow("原图",srcImage);
	imshow("均值", meanImage);
	imshow("高斯", meanImage);
	waitKey(0);
	return 0;
}


//自动匹配  需要事先选择输入点
int main5()
{
	const string basefile = "data\\1.jpg";
	const string targetfile = "data\\2.JPG";
	/*const string basefile = "data\\test\\模板.jpg";
	const string targetfile = "data\\test\\IMG_0311.JPG";*/
	const int roiWidth = 50;
	const int roiHeight = 50;
	Mat srcImage,grayImage;
	srcImage = imread(basefile);
	//转为灰度图进行处理
	cvtColor(srcImage, grayImage, CV_RGB2GRAY);
	//提取每个点周围512*512大小的ROI的SIFT特征用于比较
	vector<Point2i> inputKeyPoints;
	inputKeyPoints.push_back(Point2i(426,142));
	inputKeyPoints.push_back(Point2i(351, 217));
	inputKeyPoints.push_back(Point2i(579, 73));
	inputKeyPoints.push_back(Point2i(99, 62));
	/*inputKeyPoints.push_back(Point2i(2590,1727));
	inputKeyPoints.push_back(Point2i(153, 139));
	inputKeyPoints.push_back(Point2i(5119, 2516));
	inputKeyPoints.push_back(Point2i(83, 2590));*/

#ifdef DEBUG
	for (size_t i = 0; i < inputKeyPoints.size(); i++)
	{
		circle(srcImage, inputKeyPoints[i], 3, Scalar(0, 0, 255), 2);
	}
	namedWindow("原始测试图像", 0);
	imshow("原始测试图像", srcImage);
	waitKey(0);
#endif // DEBUG
	//设置ROI区域 尚未考虑边界问题
	vector<Point2i>::iterator it;
	vector<Mat> rois;
	for (it = inputKeyPoints.begin(); it != inputKeyPoints.end(); it++) {
		cout << "x:" << it->x << " y:" << it->y << endl;
		Rect rect(it->x - roiWidth / 2, it->y - roiHeight / 2, roiWidth, roiHeight);
		Mat roi = grayImage(rect);
		rois.push_back(roi);
		rectangle(srcImage, rect, Scalar(255, 0, 0), 1);
	}
	//提取特征
	vector<Mat> roiImageDescs;
	vector<vector<KeyPoint>> roiKeyPoints;
	
	for (size_t i = 0; i < rois.size(); i++)
	{
		vector<KeyPoint> keyPoints;
		keyPoints.reserve(100); //很容易报错
		Mat roiImageDesc;
		SiftFeatureDetector siftDetector(10);  //hessian阈值
		siftDetector.detect(rois[i], keyPoints);
		SiftDescriptorExtractor siftDescriptor;
		siftDescriptor.compute(rois[i], keyPoints, roiImageDesc);
		roiImageDescs.push_back(roiImageDesc);
		roiKeyPoints.push_back(keyPoints);

		/*FileStorage fs;
		fs.open("D:\\a.yaml", FileStorage::WRITE);
		fs << "KeyPoints" << keyPoints;
		fs << "SIFT" << roiImageDesc;
		fs.release();*/
	}

	//提取整个目标图片的sift
	//{
 		Mat dstImage = imread(targetfile);
		//灰度图转换  
		Mat dstGrayImage;
		cvtColor(dstImage, dstGrayImage, CV_RGB2GRAY);
		float ratio = 1.;
		if (dstImage.size().width>1024 || dstImage.size().height>1024)
		{
			if (dstImage.size().width>dstImage.size().height)
			{
				ratio = 1024. / dstImage.size().width;
			}
			else
			{
				ratio = 1024. / dstImage.size().height;
			}
		}
		int w = cvRound(dstGrayImage.size().width*ratio);
		int h = cvRound(dstGrayImage.size().height*ratio);
		Size size = Size(w, h);
		Mat thumbnail = Mat(size, dstGrayImage.type());
		resize(dstGrayImage, thumbnail, size);

		SiftFeatureDetector siftDetector2(1000);  //hessian阈值
		vector<KeyPoint> keyPoints2;
		siftDetector2.detect(thumbnail, keyPoints2);
		SiftDescriptorExtractor siftDescriptor2;
		Mat imageDesc;
		siftDescriptor2.compute(thumbnail, keyPoints2, imageDesc);

		//匹配
		vector<Point2f> dstMatchPoints, srcMatchPoints;
		FlannBasedMatcher matcher;
		vector<DMatch> matchePoints;
		for (size_t i = 0; i < roiImageDescs.size(); i++)
		{
			matcher.match(roiImageDescs[i], imageDesc, matchePoints, Mat());
			sort(matchePoints.begin(), matchePoints.end()); //特征点排序    
			//选择一个点作为匹配点
			Point2f dstPt(keyPoints2[matchePoints[0].trainIdx].pt.x/ratio, keyPoints2[matchePoints[0].trainIdx].pt.y / ratio);
			Point2f srcPt(inputKeyPoints[i].x - roiWidth / 2 + roiKeyPoints[i][matchePoints[0].queryIdx].pt.x, inputKeyPoints[i].y - roiHeight / 2 + roiKeyPoints[i][matchePoints[0].queryIdx].pt.y);
			dstMatchPoints.push_back(dstPt);
			srcMatchPoints.push_back(srcPt);
		}
		
#ifdef DEBUG
		//绘制匹配点
		/*for (size_t i = 0; i < 6; i++)
		{
			circle(srcImage, srcMatchPoints[i], 3, Scalar(0, 255, 0), 2);
			circle(dstImage, dstMatchPoints[i], 3, Scalar(0, 0, 255), 2);
			imshow("原始测试图像", srcImage);
			imshow("测试图像", dstImage);
			waitKey(0);
		}*/
#endif // DEBUG

		//计算H矩阵
		Mat homo = findHomography(srcMatchPoints, dstMatchPoints, CV_RANSAC);
		//对输入点进行变换
		//Point2f pt0=transform(homo, Point2f(inputKeyPoints[0]));
		Mat imageTransform1;
		warpPerspective(srcImage, imageTransform1, homo, Size(dstImage.cols, dstImage.rows));
		namedWindow("经过透视矩阵变换后", 0);
		imshow("经过透视矩阵变换后", imageTransform1);
		//circle(dstImage, pt0, 3, Scalar(0, 255, 0), 2);
		//imshow("测试图像", dstImage);
		//waitKey(0);
		waitKey(0);
	//}
	return 0;
}

//vlfeat test  亚象元分割算法
int main4()
{
	//// insert code here...
	std::cout << "Hello, World!\n";
	VL_PRINT("hello, VLFeat!\n");
	// 读入一张图片（游戏原画）   
	//Mat img = imread("data\\1.jpg");
	// 创建一个名为 "游戏原画"窗口 
	// 下面3句用于测试opencv

	//namedWindow("游戏原画");
	//imshow("游戏原画", img);
	//waitKey(3000);

	// 下面用于测试vlfeat
	cv::Mat mat = cv::imread("data\\1.jpg", CV_LOAD_IMAGE_COLOR);

	// Convert image to one-dimensional array.
	float* image = new float[mat.rows*mat.cols*mat.channels()];
	for (int i = 0; i < mat.rows; ++i) {
		for (int j = 0; j < mat.cols; ++j) {
			// Assuming three channels ...
			image[j + mat.cols*i + mat.cols*mat.rows * 0] = mat.at<cv::Vec3b>(i, j)[0];
			image[j + mat.cols*i + mat.cols*mat.rows * 1] = mat.at<cv::Vec3b>(i, j)[1];
			image[j + mat.cols*i + mat.cols*mat.rows * 2] = mat.at<cv::Vec3b>(i, j)[2];
		}
	}

	// The algorithm will store the final segmentation in a one-dimensional array.
	vl_uint32* segmentation = new vl_uint32[mat.rows*mat.cols];
	vl_size height = mat.rows;
	vl_size width = mat.cols;
	vl_size channels = mat.channels();

	// The region size defines the number of superpixels obtained.
	// Regularization describes a trade-off between the color term and the
	// spatial term.
	vl_size region = 30;
	float regularization = 1000.;
	vl_size minRegion = 10;

	vl_slic_segment(segmentation, image, width, height, channels, region, regularization, minRegion);

	// Convert segmentation.
	int** labels = new int*[mat.rows];
	for (int i = 0; i < mat.rows; ++i) {
		labels[i] = new int[mat.cols];

		for (int j = 0; j < mat.cols; ++j) {
			labels[i][j] = (int)segmentation[j + mat.cols*i];
		}
	}

	int label = 0;
	int labelTop = -1;
	int labelBottom = -1;
	int labelLeft = -1;
	int labelRight = -1;

	for (int i = 0; i < mat.rows; i++) {
		for (int j = 0; j < mat.cols; j++) {

			label = labels[i][j];

			labelTop = label;
			if (i > 0) {
				labelTop = labels[i - 1][j];
			}

			labelBottom = label;
			if (i < mat.rows - 1) {
				labelBottom = labels[i + 1][j];
			}

			labelLeft = label;
			if (j > 0) {
				labelLeft = labels[i][j - 1];
			}

			labelRight = label;
			if (j < mat.cols - 1) {
				labelRight = labels[i][j + 1];
			}

			if (label != labelTop || label != labelBottom || label != labelLeft || label != labelRight) {
				mat.at<cv::Vec3b>(i, j)[0] = 0;
				mat.at<cv::Vec3b>(i, j)[1] = 0;
				mat.at<cv::Vec3b>(i, j)[2] = 255;
			}
		}
	}

	cv::imwrite("data\\1.png", mat);
	//waitKey(6000);
	return 0;
}

//手动匹配，需要输入匹配点
int main3()
{
	//透视变换
	Mat srcImage, baseImage;
	srcImage = imread("data\\test\\data\\1.jpg");
	baseImage = imread("data\\test\\data\\BG.jpg");
	vector<Point2f> srcPoints;//配准图像上的点
	vector<Point2f> basePoints;//基准图像上的点

	//输入匹配点
	basePoints.push_back(Point2f(413, 1323));
	srcPoints.push_back(Point2f(1184, 1744));

	basePoints.push_back(Point2f(1760, 1323));
	srcPoints.push_back(Point2f(1736, 1744));

	basePoints.push_back(Point2f(1760, 2663));
	srcPoints.push_back(Point2f(1736, 2936));

	basePoints.push_back(Point2f(413, 2663));
	srcPoints.push_back(Point2f(1184, 2936));

	//获取图像1到图像2的投影映射矩阵 尺寸为3*3  
	Mat homography = findHomography(srcPoints, basePoints, CV_RANSAC);
	//透视变换   3*3的变换矩阵H  只能要求4个点
	//Mat homo = getPerspectiveTransform(srcPoints, basePoints);
	Mat imageTransform;
	warpPerspective(srcImage, imageTransform, homography, baseImage.size());

	imwrite("data\\test\\data\\result.jpg", imageTransform);
	//H变换矩阵输出
	FileStorage fs;
	fs.open("data\\test\\data\\homography.xml", FileStorage::WRITE);
	fs << "H" << homography;
	fs.release();
	Mat cvImage2 = imread("data\\test\\data\\2.jpg");
	Mat imageTransform2;
	warpPerspective(cvImage2, imageTransform2, homography, baseImage.size());
	imwrite("data\\test\\data\\result2.jpg", imageTransform2);


	namedWindow("透视变换", 0);
	imshow("透视变换", imageTransform);

	waitKey(0);

	return 0;
}


int main1() {
	//仿射变换
	//需要指定匹配点，
	//Mat dstImage_warp;
	Mat srcImage, baseImage;
	srcImage= imread("data\\test\\IMG_0311.jpg");
	baseImage = imread("data\\test\\模板.jpg");
	Point2f srcPoints[7];//配准图像上的点
    Point2f basePoints[7];//基准图像上的点
	
	basePoints[0] = Point2f(163, 3346);
	srcPoints[0] = Point2f(147, 3052);


	basePoints[1] = Point2f(5019, 3346);
	srcPoints[1] = Point2f(5810, 3045);

	basePoints[2] = Point2f(163, 109);
	srcPoints[2] = Point2f(801, 548);

	basePoints[3] = Point2f(5018, 109);
	srcPoints[3] = Point2f(5007, 633);


	/*basePoints[0] = Point2f(379, 4802);
	srcPoints[0] = Point2f(653, 3590);

	basePoints[1] = Point2f(3076, 4802);
	srcPoints[1] = Point2f(5262, 3472);


	basePoints[2] = Point2f(56, 2590);
	srcPoints[2] = Point2f(766, 2145);

	basePoints[3] = Point2f(3400, 2590);
	srcPoints[3] = Point2f(5059, 2048);

	basePoints[4] = Point2f(3346, 163);
	srcPoints[4] = Point2f(4526, 1136);

	basePoints[5] = Point2f(1728, 163);
	srcPoints[5] = Point2f(2890, 1172);

	basePoints[6] = Point2f(109, 162);
	srcPoints[6] = Point2f(1255, 1202);*/
	/*basePoints[0] = Point2f(160, 5140);
	srcPoints[0] = Point2f(95, 3780);
	
	basePoints[1] = Point2f(3303, 5136);
	srcPoints[1] = Point2f(5822, 3636);

	
	basePoints[2] = Point2f(132, 718);
	srcPoints[2] = Point2f(1188, 1354);

	basePoints[3] = Point2f(3298, 404);
	srcPoints[3] = Point2f(4529, 1131);*/

	/*basePoints[4] = Point2f(1731, 2768);
	srcPoints[4] = Point2f(3055, 2099);*/

	//
	//for (size_t i = 0; i < 4; i++)
	//{
	//	circle(baseImage, basePoints[i], 100, Scalar(0, 255, 0)); //第三个参数表示点的半径，第四个参数选择颜色。这样子我们就画出了绿色的空心点
	//}
	//namedWindow("基准图像", 0);
	//imshow("基准图像", baseImage);
	//waitKey(0);

	//for (size_t i = 0; i < 4; i++)
	//{
	//	circle(srcImage, srcPoints[i], 100, Scalar(0, 0, 255));
	//}
	//namedWindow("配准图像", 0);
	//imshow("配准图像", srcImage);
	//waitKey(0);

	//2*3的变换矩阵H
	/*Mat warpMat(2, 3, CV_32FC1);
	warpMat = getAffineTransform(srcPoints, basePoints);
	Mat dstImage;
	warpAffine(srcImage, dstImage, warpMat, baseImage.size());*/

	/*namedWindow("仿射变换", 0);
	imshow("仿射变换", dstImage);
	waitKey(0);*/

	vector<Point2f> v_srcPoints, v_basePoints;
	for (size_t i = 0; i < 4; i++)
	{
		v_srcPoints.push_back(srcPoints[i]);
		v_basePoints.push_back(basePoints[i]);
	}
	//获取图像1到图像2的投影映射矩阵 尺寸为3*3  
	Mat homo = findHomography(v_srcPoints, v_basePoints, CV_RANSAC);
	//透视变换   3*3的变换矩阵H  只能要求4个点
	//Mat homo = getPerspectiveTransform(srcPoints, basePoints);
	Mat imageTransform1;
	warpPerspective(srcImage, imageTransform1, homo, baseImage.size());

	namedWindow("透视变换", 0);
	imshow("透视变换", imageTransform1);
	waitKey(0);

	imwrite("data\\test\\result.jpg", imageTransform1);
	/*string numbers = "123";
	for (size_t i = 0; i < 3; i++)
	{
		string file;
		file += "data\\test\\IMG_";
		file +=  numbers[i];
		file += ".jpg";
		Mat img = imread(file);
		Mat resultImg;
		warpPerspective(img, resultImg, homo, baseImage.size());
		string saveFile;
		saveFile += "data\\test\\result_";
		saveFile += numbers[i];
		saveFile += ".jpg";
		imwrite(saveFile, resultImg);
	}*/


	/*FileStorage fs;
	fs.open("D:\\convert.xml", FileStorage::WRITE);
	fs << "WarpMat" << warpMat;
	fs.release();*/
	/*
	<?xml version="1.0"?>
	<opencv_storage>
	<WarpMat type_id="opencv-matrix">
	<rows>2</rows>
	<cols>3</cols>
	<dt>d</dt>
	<data>
	8.8859792143398486e-001 -2.6273934232978635e-001
	7.2171815577205479e+001 3.9890004658797701e-001
	8.2582597358581056e-001 -6.9381356307698695e+001</data></WarpMat>
	</opencv_storage>
	*/
	return 0;
}





int main0()
{

	Mat image01 = imread("data\\1.JPG");
	Mat image02 = imread("data\\2.jpg");
	namedWindow("原始测试图像",0);
	
	imshow("原始测试图像", image01);
	namedWindow("基准图像",0);
	imshow("基准图像", image02);
	waitKey(0);

	//灰度图转换  
	Mat image1, image2;
	cvtColor(image01, image1, CV_RGB2GRAY);
	cvtColor(image02, image2, CV_RGB2GRAY);

	//提取特征点    
	SurfFeatureDetector surfDetector(800);  //hessian阈值
	vector<KeyPoint> keyPoint1, keyPoint2;
	surfDetector.detect(image1, keyPoint1);
	surfDetector.detect(image2, keyPoint2);


	//特征点描述，为下边的特征点匹配做准备    
	SurfDescriptorExtractor SurfDescriptor;
	Mat imageDesc1, imageDesc2;
	SurfDescriptor.compute(image1, keyPoint1, imageDesc1);
	SurfDescriptor.compute(image2, keyPoint2, imageDesc2);


	//获得匹配特征点，并提取最优配对     
	FlannBasedMatcher matcher;
	vector<DMatch> matchePoints;
	matcher.match(imageDesc1, imageDesc2, matchePoints, Mat());
	sort(matchePoints.begin(), matchePoints.end()); //特征点排序    
	vector<Point2f> imagePoints1, imagePoints2; //获取排在前N个的最优匹配特征点 
	//vector<KeyPoint> img_keyPoints1, img_keyPoints2;
	vector<DMatch> matchePoints2;
	for (int i = 0; i<10; i++)
	{
		matchePoints2.push_back(matchePoints[i]);
		//img_keyPoints1.push_back(keyPoint1[matchePoints[i].queryIdx]);
		//img_keyPoints2.push_back(keyPoint2[matchePoints[i].trainIdx]);
		imagePoints1.push_back(keyPoint1[matchePoints[i].queryIdx].pt);
		imagePoints2.push_back(keyPoint2[matchePoints[i].trainIdx].pt);
	}
	/*SurfDescriptor.compute(image1, img_keyPoints1, imageDesc1);
	SurfDescriptor.compute(image2, img_keyPoints2, imageDesc2);
	matcher.match(imageDesc1, imageDesc2, matchePoints2, Mat());*/

	Mat imgMatches;
	drawMatches(image01, keyPoint1, image02, keyPoint2, matchePoints2, imgMatches);

	namedWindow("匹配",0);

	imshow("匹配", imgMatches);
	waitKey(0);
	//绘制特征关键点
	//Mat img_keyPoint1, img_keyPoint2;
	//drawKeypoints(image01, img_keyPoints1, img_keyPoint1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	//drawKeypoints(image02, img_keyPoints2, img_keyPoint2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

	//namedWindow("原始测试图像-特征点");

	//imshow("原始测试图像-特征点", img_keyPoint1);
	//namedWindow("基准图像-特征点");
	//imshow("基准图像-特征点", img_keyPoint2);

	/*Mat dstImage_warp;
	Point2f srcTriangle[10],dstTriangle[10];
	for (size_t i = 0; i < 10; i++)
	{
		srcTriangle[i] = imagePoints1[i];
		dstTriangle[i] = imagePoints2[i];
	}*/
	//waitKey(0);
	/*Mat warpMat(2,3,CV_32FC1);
	warpMat = getAffineTransform(srcTriangle, dstTriangle);
	FileStorage fs;
	fs.open("D:\\convert.xml",FileStorage::WRITE);
	fs << "WarpMat" << warpMat;
	fs.release();*/
	//cout << "变换矩阵为：\n" << warpMat << endl; //输出映射矩阵  
	//warpAffine(image01, dstImage_warp, warpMat, dstImage_warp.size());
	//namedWindow("仿射变换", 0);
	//imshow("仿射变换", dstImage_warp);
	//waitKey(0);
	//获取图像1到图像2的投影映射矩阵 尺寸为3*3  
	Mat homo = findHomography(imagePoints1, imagePoints2, CV_RANSAC);
	//也可以使用getPerspectiveTransform方法获得透视变换矩阵，不过要求只能有4个点，效果稍差  
	//Mat homo=getPerspectiveTransform(imagePoints1,imagePoints2);     
	//cout << "变换矩阵为：\n" << homo << endl << endl; //输出映射矩阵  
	//图像配准  
	Mat imageTransform1, imageTransform2;
	warpPerspective(image01, imageTransform1, homo, Size(image02.cols, image02.rows));
	namedWindow("经过透视矩阵变换后", 0);
	imshow("经过透视矩阵变换后", imageTransform1);

	waitKey(0);


    return 0;
}

/*
void filter_keypoint(cv::Mat &image, std::vector<cv::KeyPoint> keypoints, float minDistance, int maxCorners, float response, std::vector<cv::Point2f> &corners)
{
	//sort by response
	std::sort(keypoints.begin(), keypoints.end());

	size_t i, j, total = keypoints.size(), ncorners = 0;
	if (minDistance >= 1)
	{
		// Partition the image into larger grids
		int w = image.cols;
		int h = image.rows;

		const int cell_size = cvRound(minDistance);
		const int grid_width = (w + cell_size - 1) / cell_size;
		const int grid_height = (h + cell_size - 1) / cell_size;

		std::vector<std::vector<Point2f> > grid(grid_width*grid_height);
		minDistance *= minDistance;

		for (i = 0; i < total; i++)
		{
			int y = (int)(keypoints[i].pt.y);
			int x = (int)(keypoints[i].pt.x);

			bool good = true;

			int x_cell = x / cell_size;
			int y_cell = y / cell_size;

			int x1 = x_cell - 1;
			int y1 = y_cell - 1;
			int x2 = x_cell + 1;
			int y2 = y_cell + 1;

			// boundary check
			x1 = std::max(0, x1);
			y1 = std::max(0, y1);
			x2 = std::min(grid_width - 1, x2);
			y2 = std::min(grid_height - 1, y2);

			for (int yy = y1; yy <= y2; yy++)
			{
				for (int xx = x1; xx <= x2; xx++)
				{
					vector <Point2f> &m = grid[yy*grid_width + xx];

					if (m.size())
					{
						for (j = 0; j < m.size(); j++)
						{
							float dx = x - m[j].x;
							float dy = y - m[j].y;

							if (dx*dx + dy*dy < minDistance)
							{
								good = false;
								goto break_out;
							}
						}
					}
				}
			}

		break_out:

			if (good)
			{

				if (keypoints[i].response >= response)
				{
					// printf("%d: %d %d -> %d %d, %d, %d -- %d %d %d %d, %d %d, c=%d\n",
					//    i,x, y, x_cell, y_cell, (int)minDistance, cell_size,x1,y1,x2,y2, grid_width,grid_height,c);
					grid[y_cell*grid_width + x_cell].push_back(Point2f((float)x, (float)y));
					cout << "response:" << keypoints[i].response << endl;
					corners.push_back(Point2f((float)x, (float)y));
				}
				++ncorners;
				if (maxCorners > 0 && (int)ncorners == maxCorners)
					break;
			}
		}
	}
	else
	{
		for (i = 0; i < total; i++)
		{
			int y = (int)(keypoints[i].pt.y);
			int x = (int)(keypoints[i].pt.x);

			corners.push_back(Point2f((float)x, (float)y));
			++ncorners;
			if (maxCorners > 0 && (int)ncorners == maxCorners)
				break;
		}
	}
}

*/

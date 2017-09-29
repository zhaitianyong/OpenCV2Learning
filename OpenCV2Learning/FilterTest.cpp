/************************************************************************
> File Name: 
> Author:atway
> Mail:atway#126.com(#=>@)
> Created Time: 2014年10月15日 星期三 12时00分33秒
************************************************************************/



//各种滤波器练习

#include "stdafx.h"
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>  
#include <opencv2\imgproc\imgproc.hpp>
#include <iostream>  

using namespace std;
using namespace cv;

//全局变量
Mat srcImage, dstboxImage, dstmeanImage, dstGaussianImage,dstMedianImage,dstBilateralImage;
int g_nBoxFilterValue = 3;
int g_nMeanFilterValue = 3;
int g_nGaussanianFilterValue = 3;
int g_nMedianFilterValue = 3;
int g_nBilateralFilterValue = 3;
void on_BoxFilter(int, void *);
void on_MeanFilter(int, void *);
void on_GaussaianFilter(int, void *);
void on_MedianFilter(int,void *);
void on_BilateralFilter(int, void *); 

void main_filter()
{
 
	srcImage = imread("data\\1.jpg");

	namedWindow("原图窗口");
	imshow("原图窗口",srcImage);

	waitKey(1000);

	while (1)
	{
		char cmd;
		cout << "请输入要滤波的方法:" << endl
			<< "a -------- 方框滤波" << endl
			<< "b -------- 均值滤波" << endl
			<< "c -------- 高斯滤波" << endl
			<< "d -------- 中值滤波" << endl
			<< "e -------- 双边滤波" << endl
			<< "q -------- 退出" << endl;
		cin >> cmd;

		if (cmd=='q')
		{
			break;
		}
		switch (cmd)
		{
		case 'a':
		{
			//----------------方框滤波器----------------------------//
			dstboxImage = srcImage.clone();
			namedWindow("方框滤波");
			createTrackbar("内核值：", "方框滤波", &g_nBoxFilterValue, 10, on_BoxFilter);
			on_BoxFilter(0, 0);
			waitKey(0);
			destroyWindow("方框滤波");
		}
		break;
		case 'b':
		{
			//----------------均值滤波器----------------------------//
			dstmeanImage = srcImage.clone();
			namedWindow("均值滤波");
			createTrackbar("内核值：", "均值滤波", &g_nMeanFilterValue, 10, on_MeanFilter);
			on_MeanFilter(0, 0);
			waitKey(0);
			destroyWindow("均值滤波");
		}
		break;
		case 'c':
		{
			//----------------高斯滤波器----------------------------//
			dstGaussianImage = srcImage.clone();
			namedWindow("高斯滤波");
			createTrackbar("内核值：", "高斯滤波", &g_nGaussanianFilterValue, 10, on_GaussaianFilter);
			on_GaussaianFilter(0, 0);
			waitKey(0);
			destroyWindow("高斯滤波");
		}break;
		case 'd':
		{
			//------------------中值滤波器-----------------------------//
			dstMedianImage = srcImage.clone();
			namedWindow("中值滤波器");
			createTrackbar("内核值", "中值滤波器", &g_nMedianFilterValue, 10, on_MedianFilter);
			on_MedianFilter(0, 0);
			waitKey(0);
			destroyWindow("中值滤波器");
		}
		break;
		case 'e':
		{
			//------------------双边滤波器-----------------------------//
			dstBilateralImage = srcImage.clone();
			namedWindow("双边滤波器");
			createTrackbar("内核值", "双边滤波器", &g_nBilateralFilterValue, 100, on_BilateralFilter);
			on_BilateralFilter(0, 0);
			waitKey(0);
			destroyWindow("双边滤波器");
		}
		break;
		default:
			break;
		}
	}
	//waitKey(0);
}

void on_BoxFilter(int, void *)
{
	boxFilter(srcImage, dstboxImage, -1, Size(g_nBoxFilterValue + 1, g_nBoxFilterValue + 1));
	imshow("方框滤波",dstboxImage);
}

void on_MeanFilter(int, void *)
{
	blur(srcImage, dstmeanImage, Size(g_nMeanFilterValue + 1, g_nMeanFilterValue + 1));
	imshow("均值滤波", dstmeanImage);
}

void on_GaussaianFilter(int, void *)
{
	GaussianBlur(srcImage, dstGaussianImage, Size(g_nGaussanianFilterValue*2+1, g_nGaussanianFilterValue*2+1), 0, 0);
	imshow("高斯滤波", dstGaussianImage);
}

void on_MedianFilter(int, void *)
{
	medianBlur(srcImage, dstMedianImage,g_nMedianFilterValue*2+1);
	imshow("中值滤波器", dstMedianImage);
}

void on_BilateralFilter(int, void *)
{
	bilateralFilter(srcImage, dstBilateralImage, g_nBilateralFilterValue , g_nBilateralFilterValue*2, g_nBilateralFilterValue/2);
	imshow("双边滤波器", dstBilateralImage);
}

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
#include <iostream>  


using namespace std;
using namespace cv;

void main_Resize()
{
	Mat srcImage, dilateImage;
	srcImage = imread("data\\BG.jpg");
	namedWindow("原图");
	imshow("原图", srcImage);
	waitKey(2000);
	Mat element = getStructuringElement(MORPH_RECT, Size(8, 8));
	morphologyEx(srcImage, dilateImage, MORPH_DILATE, element);
	//imwrite("data\\dilateImage.jpg", dilateImage);
	imshow("膨胀效果图", dilateImage);
	waitKey(100);
	Mat erodeImage;
	morphologyEx(srcImage, erodeImage, MORPH_ERODE, element);
	imshow("腐蚀效果图", erodeImage);
	waitKey(0);
	//imwrite("data\\erodeImage.jpg", erodeImage);
	Mat openImage; //先腐蚀（erode） 后膨胀(dilate)
	morphologyEx(erodeImage, openImage, MORPH_DILATE, element);
	imshow("开运算效果图", openImage); //放大黑洞和噪声点
	waitKey(0);
	//imwrite("data\\openImage.jpg", openImage);
	Mat closeImage; //先膨胀 后腐蚀
	morphologyEx(dilateImage, closeImage, MORPH_ERODE, element);
	imshow("闭运算效果图", closeImage); //去除黑洞 噪声
	//imwrite("data\\closeImage.jpg",closeImage);
	waitKey(0);
	Mat gradientImage = dilateImage - erodeImage;

	imshow("形态学梯度效果图", gradientImage); 
	//imwrite("data\\gradientImage.jpg", gradientImage);
	waitKey(0);
	Mat tophatImage = srcImage - openImage;
	imshow("顶帽效果图", tophatImage);
	//imwrite("data\\tophatImage.jpg", tophatImage);

	Mat blackhatImage = closeImage - srcImage;
	imshow("黑帽效果图", blackhatImage);
	//imwrite("data\\blackhatImage.jpg", blackhatImage);

	Mat black_white;
	addWeighted(tophatImage, 0.5, blackhatImage, 0.5, 0, black_white);

	imshow("black_white", tophatImage);
	waitKey(0);
}

void main_pry()
{
	Mat srcImage, dstImage;
	srcImage= imread("data\\1.jpg");
	namedWindow("原图");
	imshow("原图",srcImage);

	waitKey(2000);
	//降采样
	pyrDown(srcImage, dstImage, Size(srcImage.cols / 2, srcImage.rows / 2));
	imshow("放缩",dstImage);
	waitKey(2000);
	Mat dstUpImage;
	//向上采样
	pyrUp(dstImage, dstUpImage, srcImage.size());
	imshow("恢复",dstUpImage);
	waitKey(2000);
	//原图-模糊图像
	Mat diffImage = srcImage - dstUpImage;
	imshow("作差",diffImage);
	waitKey(0);
 
	system("pause");
}
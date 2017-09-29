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
#include <opencv2\calib3d\calib3d.hpp>
#include <iostream>  
#include "improcess.h"
using namespace std;
using namespace cv;

//模板匹配 计算出中心点的像素坐标
/// 全局变量
Mat img; Mat templ; Mat result;
char* image_window = "Source Image";
char* result_window = "Result window";

int match_method;
int max_Trackbar = 5;

/// 函数声明
void MatchingMethod(int, void*);

/** @主函数 */
int main_match(int argc, char** argv)
{
	/// 载入原图像和模板块
	/*img = imread(argv[1], 1);
	templ = imread(argv[2], 1);*/
	img = imread("data\\center\\BG.jpg");
	//templ = imread("data\\center\\mask.jpg");
	/// 创建窗口
	/*namedWindow(image_window, CV_WINDOW_AUTOSIZE);
	namedWindow(result_window, CV_WINDOW_AUTOSIZE);*/
	namedWindow(image_window,CV_WINDOW_NORMAL);
	namedWindow(result_window, CV_WINDOW_NORMAL);
	/// 创建滑动条
	char* trackbar_label = "Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED";
	createTrackbar(trackbar_label, image_window, &match_method, max_Trackbar, MatchingMethod);

	MatchingMethod(0, 0);

	waitKey(0);
	return 0;
}


/**
* @函数 MatchingMethod
* @简单的滑动条回调函数
*/
void MatchingMethod(int, void*)
{
	/// 将被显示的原图像
	Mat img_display;
	img.copyTo(img_display);

	//需要注意的问题是 分辨率需要统一
	//方式 提取大棋盘格和模板像元大小
	//统计每个格子像元大小或者面积
	
	/// 创建输出结果的矩阵
	int result_cols = img.cols - templ.cols + 1;
	int result_rows = img.rows - templ.rows + 1;

	result.create(result_cols, result_rows, CV_32FC1);

	/// 进行匹配和标准化
	matchTemplate(img, templ, result, match_method);
	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

	/// 通过函数 minMaxLoc 定位最匹配的位置
	double minVal; double maxVal; Point minLoc; Point maxLoc;
	Point matchLoc;

	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

	/// 对于方法 SQDIFF 和 SQDIFF_NORMED, 越小的数值代表更高的匹配结果. 而对于其他方法, 数值越大匹配越好
	if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED)
	{
		matchLoc = minLoc;
	}
	else
	{
		matchLoc = maxLoc;
	}

	//让我看看您的最终结果
	rectangle(img_display, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar(0, 0, 255), 4, 8, 0);
	circle(img_display, Point(matchLoc.x + templ.cols / 2, matchLoc.y + templ.rows / 2),4, Scalar(0, 255, 0),8);
	rectangle(result, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar(0,0,255), 2, 8, 0);

	imshow(image_window, img_display);
	imshow(result_window, result);

	return;
}
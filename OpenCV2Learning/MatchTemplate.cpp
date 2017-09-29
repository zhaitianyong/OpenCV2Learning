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
#include <opencv2\calib3d\calib3d.hpp>
#include <iostream>  
#include "improcess.h"
using namespace std;
using namespace cv;

//ģ��ƥ�� ��������ĵ����������
/// ȫ�ֱ���
Mat img; Mat templ; Mat result;
char* image_window = "Source Image";
char* result_window = "Result window";

int match_method;
int max_Trackbar = 5;

/// ��������
void MatchingMethod(int, void*);

/** @������ */
int main_match(int argc, char** argv)
{
	/// ����ԭͼ���ģ���
	/*img = imread(argv[1], 1);
	templ = imread(argv[2], 1);*/
	img = imread("data\\center\\BG.jpg");
	//templ = imread("data\\center\\mask.jpg");
	/// ��������
	/*namedWindow(image_window, CV_WINDOW_AUTOSIZE);
	namedWindow(result_window, CV_WINDOW_AUTOSIZE);*/
	namedWindow(image_window,CV_WINDOW_NORMAL);
	namedWindow(result_window, CV_WINDOW_NORMAL);
	/// ����������
	char* trackbar_label = "Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED";
	createTrackbar(trackbar_label, image_window, &match_method, max_Trackbar, MatchingMethod);

	MatchingMethod(0, 0);

	waitKey(0);
	return 0;
}


/**
* @���� MatchingMethod
* @�򵥵Ļ������ص�����
*/
void MatchingMethod(int, void*)
{
	/// ������ʾ��ԭͼ��
	Mat img_display;
	img.copyTo(img_display);

	//��Ҫע��������� �ֱ�����Ҫͳһ
	//��ʽ ��ȡ�����̸��ģ����Ԫ��С
	//ͳ��ÿ��������Ԫ��С�������
	
	/// �����������ľ���
	int result_cols = img.cols - templ.cols + 1;
	int result_rows = img.rows - templ.rows + 1;

	result.create(result_cols, result_rows, CV_32FC1);

	/// ����ƥ��ͱ�׼��
	matchTemplate(img, templ, result, match_method);
	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

	/// ͨ������ minMaxLoc ��λ��ƥ���λ��
	double minVal; double maxVal; Point minLoc; Point maxLoc;
	Point matchLoc;

	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

	/// ���ڷ��� SQDIFF �� SQDIFF_NORMED, ԽС����ֵ������ߵ�ƥ����. ��������������, ��ֵԽ��ƥ��Խ��
	if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED)
	{
		matchLoc = minLoc;
	}
	else
	{
		matchLoc = maxLoc;
	}

	//���ҿ����������ս��
	rectangle(img_display, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar(0, 0, 255), 4, 8, 0);
	circle(img_display, Point(matchLoc.x + templ.cols / 2, matchLoc.y + templ.rows / 2),4, Scalar(0, 255, 0),8);
	rectangle(result, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar(0,0,255), 2, 8, 0);

	imshow(image_window, img_display);
	imshow(result_window, result);

	return;
}
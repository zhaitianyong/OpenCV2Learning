/************************************************************************
> File Name: 
> Author:atway
> Mail:atway#126.com(#=>@)
> Created Time: 2014��10��15�� ������ 12ʱ00��33��
************************************************************************/



//�����˲�����ϰ

#include "stdafx.h"
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>  
#include <opencv2\imgproc\imgproc.hpp>
#include <iostream>  

using namespace std;
using namespace cv;

//ȫ�ֱ���
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

	namedWindow("ԭͼ����");
	imshow("ԭͼ����",srcImage);

	waitKey(1000);

	while (1)
	{
		char cmd;
		cout << "������Ҫ�˲��ķ���:" << endl
			<< "a -------- �����˲�" << endl
			<< "b -------- ��ֵ�˲�" << endl
			<< "c -------- ��˹�˲�" << endl
			<< "d -------- ��ֵ�˲�" << endl
			<< "e -------- ˫���˲�" << endl
			<< "q -------- �˳�" << endl;
		cin >> cmd;

		if (cmd=='q')
		{
			break;
		}
		switch (cmd)
		{
		case 'a':
		{
			//----------------�����˲���----------------------------//
			dstboxImage = srcImage.clone();
			namedWindow("�����˲�");
			createTrackbar("�ں�ֵ��", "�����˲�", &g_nBoxFilterValue, 10, on_BoxFilter);
			on_BoxFilter(0, 0);
			waitKey(0);
			destroyWindow("�����˲�");
		}
		break;
		case 'b':
		{
			//----------------��ֵ�˲���----------------------------//
			dstmeanImage = srcImage.clone();
			namedWindow("��ֵ�˲�");
			createTrackbar("�ں�ֵ��", "��ֵ�˲�", &g_nMeanFilterValue, 10, on_MeanFilter);
			on_MeanFilter(0, 0);
			waitKey(0);
			destroyWindow("��ֵ�˲�");
		}
		break;
		case 'c':
		{
			//----------------��˹�˲���----------------------------//
			dstGaussianImage = srcImage.clone();
			namedWindow("��˹�˲�");
			createTrackbar("�ں�ֵ��", "��˹�˲�", &g_nGaussanianFilterValue, 10, on_GaussaianFilter);
			on_GaussaianFilter(0, 0);
			waitKey(0);
			destroyWindow("��˹�˲�");
		}break;
		case 'd':
		{
			//------------------��ֵ�˲���-----------------------------//
			dstMedianImage = srcImage.clone();
			namedWindow("��ֵ�˲���");
			createTrackbar("�ں�ֵ", "��ֵ�˲���", &g_nMedianFilterValue, 10, on_MedianFilter);
			on_MedianFilter(0, 0);
			waitKey(0);
			destroyWindow("��ֵ�˲���");
		}
		break;
		case 'e':
		{
			//------------------˫���˲���-----------------------------//
			dstBilateralImage = srcImage.clone();
			namedWindow("˫���˲���");
			createTrackbar("�ں�ֵ", "˫���˲���", &g_nBilateralFilterValue, 100, on_BilateralFilter);
			on_BilateralFilter(0, 0);
			waitKey(0);
			destroyWindow("˫���˲���");
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
	imshow("�����˲�",dstboxImage);
}

void on_MeanFilter(int, void *)
{
	blur(srcImage, dstmeanImage, Size(g_nMeanFilterValue + 1, g_nMeanFilterValue + 1));
	imshow("��ֵ�˲�", dstmeanImage);
}

void on_GaussaianFilter(int, void *)
{
	GaussianBlur(srcImage, dstGaussianImage, Size(g_nGaussanianFilterValue*2+1, g_nGaussanianFilterValue*2+1), 0, 0);
	imshow("��˹�˲�", dstGaussianImage);
}

void on_MedianFilter(int, void *)
{
	medianBlur(srcImage, dstMedianImage,g_nMedianFilterValue*2+1);
	imshow("��ֵ�˲���", dstMedianImage);
}

void on_BilateralFilter(int, void *)
{
	bilateralFilter(srcImage, dstBilateralImage, g_nBilateralFilterValue , g_nBilateralFilterValue*2, g_nBilateralFilterValue/2);
	imshow("˫���˲���", dstBilateralImage);
}

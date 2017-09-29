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
#include <iostream>  


using namespace std;
using namespace cv;

void main_Resize()
{
	Mat srcImage, dilateImage;
	srcImage = imread("data\\BG.jpg");
	namedWindow("ԭͼ");
	imshow("ԭͼ", srcImage);
	waitKey(2000);
	Mat element = getStructuringElement(MORPH_RECT, Size(8, 8));
	morphologyEx(srcImage, dilateImage, MORPH_DILATE, element);
	//imwrite("data\\dilateImage.jpg", dilateImage);
	imshow("����Ч��ͼ", dilateImage);
	waitKey(100);
	Mat erodeImage;
	morphologyEx(srcImage, erodeImage, MORPH_ERODE, element);
	imshow("��ʴЧ��ͼ", erodeImage);
	waitKey(0);
	//imwrite("data\\erodeImage.jpg", erodeImage);
	Mat openImage; //�ȸ�ʴ��erode�� ������(dilate)
	morphologyEx(erodeImage, openImage, MORPH_DILATE, element);
	imshow("������Ч��ͼ", openImage); //�Ŵ�ڶ���������
	waitKey(0);
	//imwrite("data\\openImage.jpg", openImage);
	Mat closeImage; //������ ��ʴ
	morphologyEx(dilateImage, closeImage, MORPH_ERODE, element);
	imshow("������Ч��ͼ", closeImage); //ȥ���ڶ� ����
	//imwrite("data\\closeImage.jpg",closeImage);
	waitKey(0);
	Mat gradientImage = dilateImage - erodeImage;

	imshow("��̬ѧ�ݶ�Ч��ͼ", gradientImage); 
	//imwrite("data\\gradientImage.jpg", gradientImage);
	waitKey(0);
	Mat tophatImage = srcImage - openImage;
	imshow("��ñЧ��ͼ", tophatImage);
	//imwrite("data\\tophatImage.jpg", tophatImage);

	Mat blackhatImage = closeImage - srcImage;
	imshow("��ñЧ��ͼ", blackhatImage);
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
	namedWindow("ԭͼ");
	imshow("ԭͼ",srcImage);

	waitKey(2000);
	//������
	pyrDown(srcImage, dstImage, Size(srcImage.cols / 2, srcImage.rows / 2));
	imshow("����",dstImage);
	waitKey(2000);
	Mat dstUpImage;
	//���ϲ���
	pyrUp(dstImage, dstUpImage, srcImage.size());
	imshow("�ָ�",dstUpImage);
	waitKey(2000);
	//ԭͼ-ģ��ͼ��
	Mat diffImage = srcImage - dstUpImage;
	imshow("����",diffImage);
	waitKey(0);
 
	system("pause");
}
/************************************************************************
> File Name: 
> Author:atway
> Mail:atway#126.com(#=>@)
> Created Time: 2014��10��15�� ������ 12ʱ00��33��
************************************************************************/

#include "stdafx.h"
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>  
#include <opencv2\nonfree\nonfree.hpp>  
#include <opencv2\legacy\legacy.hpp>
#include <iostream>  
#include <string.h>
#include <math.h>
extern "C" {
#include "vl/generic.h"
#include <vl/stringop.h>
#include <vl/sift.h>
#include <vl/getopt_long.h>
//#include "vl/slic.h"
}
#define PI 3.1415926 
using namespace cv;
using namespace std;

void drawArrow(cv::Mat& img, cv::Point2f pStart, cv::Point2f pEnd, int len, int alpha, cv::Scalar& color, int thickness=1, int lineType=8)
{
	//const double PI = 3.1415926;
	Point arrow;
	//���� �� �ǣ���򵥵�һ�����������ͼʾ���Ѿ�չʾ���ؼ����� atan2 ��������������棩   
	double angle = atan2((double)(pStart.y - pEnd.y), (double)(pStart.x - pEnd.x));

	line(img, pStart, pEnd, color, thickness, lineType);

	//������Ǳߵ���һ�˵Ķ˵�λ�ã�����Ļ��������Ҫ����ͷ��ָ��Ҳ����pStart��pEnd��λ�ã� 
	arrow.x = pEnd.x + len * cos(angle + PI * alpha / 180);

	arrow.y = pEnd.y + len * sin(angle + PI * alpha / 180);

	line(img, pEnd, arrow, color, thickness, lineType);

	arrow.x = pEnd.x + len * cos(angle - PI * alpha / 180);

	arrow.y = pEnd.y + len * sin(angle - PI * alpha / 180);

	line(img, pEnd, arrow, color, thickness, lineType);
}

void getkeyPoints2(Mat& img, vector<KeyPoint>& keyPoint, Mat& imageDesc)
{
	SiftFeatureDetector siftDetector(0, 5);// siftDetector(0, 3, 0.04); //Ĭ��0.04
	SiftDescriptorExtractor SiftDescriptor;
	cout << "������ȡ������" << endl;
	siftDetector.detect(img, keyPoint);
	cout << "��������������" << endl;
	SiftDescriptor.compute(img, keyPoint, imageDesc);
}

int main8888(int argc, const char * argv[]) {

	//cout << sin(30) << endl;
	//cout << sin(PI * 30 / 180) << endl;

	VL_PRINT("Hello world!\n");
	// beaver.png  C:\\vlfeat-0.9.20\\data\\roofs1.jpg
	//load image
	Mat srcImage, grayImage, tmpImage;
	srcImage = imread("data\\beaver.png",CV_LOAD_IMAGE_COLOR);

	//�Ҷ�ͼ
	cvtColor(srcImage, grayImage, CV_RGB2BGR);
	tmpImage = srcImage.clone();

	// �˴������������Ķ��忴����vl_sift_new�����еĽ���
	int noctaves = 4, nlevels = 5, o_min = 0;
	//��ζ�ԭͼ���зֲ�
	noctaves=log2(min(grayImage.cols, grayImage.rows));


	// vl_sift_pix ����float������
	vl_sift_pix *imgdata = new vl_sift_pix[grayImage.rows * grayImage.cols];

	// ��ԭͼ���Ƶ�float�͵�vl_sift_pix������
	unsigned char *Pixel;
	for (int i = 0; i<grayImage.rows; i++)
	{
		uchar *Pixel = grayImage.ptr<uchar>(i);
		for (int j = 0; j<grayImage.cols; j++)
		{
			//Pixel = (unsigned char*)(img->imageData + i*img->width + j);
			imgdata[i*grayImage.cols + j] = Pixel[j];
		}
	}

	// VlSiftFilt: This filter implements the SIFT detector and descriptor.
	// ���������ʵ����SIFT�������������
	VlSiftFilt *siftfilt = NULL;

	// ����һ���µ�sift�˲���  
	siftfilt = vl_sift_new(grayImage.cols, grayImage.rows, noctaves, nlevels, o_min);

	vector<KeyPoint> keyPoints;
	//vector<Mat>  descriptor;

	int keypoint = 0;
	int idx_point = 0;             //������ĸ���
	int idx_descri = 0;            //�������������ĸ��� >= idx_point
	int noctaves_ct = 0;
	if (vl_sift_process_first_octave(siftfilt, imgdata) != VL_ERR_EOF)
	{
		while (1)
		{
			int _w=vl_sift_get_octave_width(siftfilt);
			int _h = vl_sift_get_octave_height(siftfilt);
			cout << "octave" << endl;
			cout << "width:" << _w << " height:" << _h << endl;
			//����ÿ���еĹؼ���  
			vl_sift_detect(siftfilt);
			//����������ÿ����  
			keypoint += siftfilt->nkeys;//��⵽�Ĺؼ������Ŀ
			cout << "�ؼ�����Ŀ��" << endl << keypoint << endl;

			VlSiftKeypoint *pKeyPoint = siftfilt->keys;//��⵽�Ĺؼ���
			for (int i = 0; i<siftfilt->nkeys; i++)
			{
				VlSiftKeypoint temptKeyPoint = *pKeyPoint;
				//circle(srcImage, cvPoint(temptKeyPoint.x, temptKeyPoint.y), temptKeyPoint.sigma / 2, CV_RGB(255, 0, 0));
				KeyPoint kp(temptKeyPoint.x, temptKeyPoint.y,temptKeyPoint.sigma,temptKeyPoint.s, 0, noctaves_ct);
				keyPoints.push_back(kp);
				//idx_point++;
				//���㲢����ÿ����ķ���  
				double angles[4];
				int angleCount = vl_sift_calc_keypoint_orientations(siftfilt, angles, &temptKeyPoint);//����ؼ���ķ���
				for (int j = 0; j<angleCount; j++)
				{
					double temptAngle = angles[j];
					//printf("%d: %f\n", j, temptAngle);
					//����ÿ�����������  
					float *descriptors = new float[128];
					vl_sift_calc_keypoint_descriptor(siftfilt, descriptors, &temptKeyPoint, temptAngle);
					/*int k = 0;
					while (k<128)
					{
						printf("%d: %f", k, descriptors[k]);
						printf("; ");
						k++;
					}

					printf("\n");*/
					delete[]descriptors;
					descriptors = NULL;
				}

				pKeyPoint++;
			}
			
			//��һ��  
			if (vl_sift_process_next_octave(siftfilt) == VL_ERR_EOF)
			{
				break;
			}
			noctaves_ct++;
			//free(pKeyPoint);  
			keypoint = 0;
		}
	}
	vl_sift_delete(siftfilt);
	for (size_t i = 0; i < keyPoints.size(); i++)
	{
		//��һ�������
		//if (keyPoints[i].octave==0)
		//{
		//	circle(srcImage, keyPoints[i].pt, keyPoints[i].size / 2, CV_RGB(255, 0, 0));
		//	//line(srcImage,)
		//	cout << "�Ƕȣ�" << keyPoints[i].angle << " �뾶��" << keyPoints[i].size << endl;
		//	
		//}
		//�ڶ��������
		/*if (keyPoints[i].octave == 1)
		{
			circle(srcImage, keyPoints[i].pt, keyPoints[i].size / 2, CV_RGB(0, 255, 0));
		}*/
		//�����������
		/*if (keyPoints[i].octave == 2)
		{
			circle(srcImage, keyPoints[i].pt, keyPoints[i].size / 2, CV_RGB(0, 0, 255));
		}*/
		//���Ĳ������
		/*if (keyPoints[i].octave == 3)
		{*/
			circle(srcImage, keyPoints[i].pt, keyPoints[i].size / 2, CV_RGB(255, 0, 0));
			//���㼫����
			Point2f endPt;
			endPt.x = (keyPoints[i].size / 2)*cos(keyPoints[i].angle)+ keyPoints[i].pt.x;
			endPt.y = (keyPoints[i].size / 2)*sin(keyPoints[i].angle)+keyPoints[i].pt.y;
			//line(srcImage,keyPoints[i].pt,endPt,cv::Scalar::all(-1));
			drawArrow(srcImage, keyPoints[i].pt, endPt, 2, 30, Scalar::all(-1));
		//}
		//circle(srcImage, keyPoints[i].pt, keyPoints[i].size / 2, CV_RGB(125, 125, 125));
	}
	

	imshow("ԭͼ2", tmpImage);

	waitKey(0);



	vector<KeyPoint> cvKeyPoints;
	Mat imageDesc;
	getkeyPoints2(tmpImage,cvKeyPoints,imageDesc);
   
	for (size_t i = 0; i < cvKeyPoints.size(); i++)
	{
		//circle(tmpImage, cvKeyPoints[i].pt, cvKeyPoints[i].size, CV_RGB(255, 0, 0));
		//���㼫����
		Point2f endPt;
		endPt.x = (cvKeyPoints[i].size )*cos(2 * PI- cvKeyPoints[i].angle) + cvKeyPoints[i].pt.x;
		endPt.y = (cvKeyPoints[i].size )*sin(2 * PI-cvKeyPoints[i].angle) + cvKeyPoints[i].pt.y;
		//line(srcImage,keyPoints[i].pt,endPt,cv::Scalar::all(-1));
		drawArrow(tmpImage, cvKeyPoints[i].pt, endPt, 2, 30, Scalar(0, 0, 255));
	}
	 
	imshow("ԭͼ2", tmpImage);

	waitKey(0);


	//system("pause");
	return 0;
}

//C:/gtk_2.24.10_win32/include/gtk-2.0
//C:/gtk_2.24.10_win32/lib/gtk-2.0/include
//C:/gtk_2.24.10_win32/include/atk-1.0
//C:/gtk_2.24.10_win32/include/cairo
//C:/gtk_2.24.10_win32/include/gdk-pixbuf-2.0
//C:/gtk_2.24.10_win32/include/pango-1.0
//C:/gtk_2.24.10_win32/include/glib-2.0
//C:/gtk_2.24.10_win32/lib/glib-2.0/include
//C:/gtk_2.24.10_win32/include
//C:/gtk_2.24.10_win32/include/freetype2
//C:/gtk_2.24.10_win32/include/libpng14
#include "stdafx.h"
#include "improcess.h"
#include <iostream>

namespace atway
{
	using namespace std;
	using namespace cv;
	//--------------------------------------------------
	//��ȡ1mm ����Ԫԭ�����������㵥����Ԫ��С
	//srcImage  ����ͼ��
	//pixelSize �����Ԫ��С
	//thresh ��������Ԫ���� 0-1 ֮�� Ĭ��Ϊ0.5
	//direction 0 ��ʾˮƽ 1��ʾ��ֱ
	//----------------------------------------------
	void measure_ruler_edge(const Mat& srcImage, double& pixelSize, double thresh, int direction)
	{
		if (!srcImage.data)
		{
			cout << "data load failed" << endl;
			return;
		}
		if (thresh < 0) thresh = 0.1;

		if (thresh > 1) thresh = 1.;

		Mat grayImage;
		//ת�ɻҶ�ͼ��
		cvtColor(srcImage, grayImage, CV_RGB2GRAY);
		//ƽ��ȥ���� 
		Mat smoothImage;
		GaussianBlur(grayImage, smoothImage, Size(5, 5), 0, 0);
		//ͼ���ֵ��
		Mat binaryImage;
		threshold(smoothImage, binaryImage, 0, 255, THRESH_OTSU);
		threshold(binaryImage, binaryImage, 0, 255, THRESH_BINARY_INV);
		//ͶӰ�ָ�
		vector<int> v_lines;
		int pthresh = direction == 0 ? cvRound(binaryImage.cols*thresh) : cvRound(binaryImage.rows*thresh);
		projectionSegmentation(binaryImage, v_lines, pthresh, 0, false); //x����0  y����1
		//������Ԫ�ߴ�
		if (v_lines.size() < 3)
		{
			cout << "ͼƬ�е�ֱ����ȡ����3��" << endl;
			return;
		}
		//ͳ�Ƹ���  �׵���һ��
		vector<int> vpt;
		for (size_t i = 1; i < v_lines.size() - 1; i++)
		{
			vpt.push_back((v_lines[i + 1] - v_lines[i]));
		}
		//����
		sort(vpt.begin(), vpt.end());
		//ѡ����λ��
		int count = vpt[vpt.size() / 2];
		cout << "pixels count:" << count << endl;
		pixelSize = 1. / count;
	}

	void projectionSegmentation(const Mat& src, vector<int>& lines, int thresh, int direction, bool isBoth) {
		vector<int> data; //����
		 //ˮƽ����
		if (direction == 0) {
			for (int i = 0; i < src.rows; i++) {
				int itmp = countNonZero(src.row(i));
				data.push_back(itmp);
			}
		}//��ֱ����
		else {

			for (int i = 0; i < src.cols; i++) {
				int itmp = countNonZero(src.col(i));
				data.push_back(itmp);
			}
		}
		//ˮƽ����
		if (direction == 0)
		{
			for (int i = 0; i < src.rows - 1; i++)
			{
				if (isBoth)
				{
					if ((data[i] < thresh && data[i + 1] >= thresh) || (data[i] >= thresh && data[i + 1] < thresh))
					{
						lines.push_back(i);
					}
				}
				else {
					if ((data[i] < thresh && data[i + 1] >= thresh))
					{
						lines.push_back(i);
					}
				}

			}
		}
		else {
			//��ֱ����

			for (int i = 0; i < src.cols - 1; i++)
			{
				if (isBoth)
				{
					if ((data[i] >= thresh && data[i + 1] < thresh) || (data[i] < thresh && data[i + 1] >= thresh))
					{
						lines.push_back(i);
					}
				}
				else
				{
					if ((data[i] >= thresh && data[i + 1] < thresh))
					{
						lines.push_back(i);
					}
				}
			}
		}
	}

	//--------------------------------------------------
	//��ȡ1mm ����Ԫԭ�����������㵥����Ԫ��С
	//srcImage  ����ͼ��
	//pixelSize �����Ԫ��С
	//thresh ��������Ԫ���� 0-1 ֮�� Ĭ��Ϊ0.5
	//direction 0 ��ʾˮƽ 1��ʾ��ֱ
	//----------------------------------------------
	void measure_ruler_center(const Mat& srcImage, double& pixelSize, double thresh, int direction)
	{
		Mat grayImage;
		if (!srcImage.data)
		{
			cout << "data load failed" << endl;
			return;
		}

		if (thresh < 0) thresh = 0.1;

		if (thresh > 1) thresh = 1.;

		//�Ҷ�ͼ����
		cvtColor(srcImage, grayImage, CV_RGB2GRAY);
		//ƽ��ȥ���� 
		Mat smoothImage;
		GaussianBlur(grayImage, smoothImage, Size(5, 5), 0, 0);
		//ͼ��Resize���� 
		int ratio = 4;
		Mat thumbnailImage(Size(grayImage.cols / ratio, grayImage.rows / ratio), smoothImage.type(), cv::Scalar(0));
		resize(smoothImage, thumbnailImage, Size(grayImage.cols / 4, grayImage.rows / 4));
		//��ֵ��
		Mat binaryImage;
		threshold(thumbnailImage, binaryImage, 0, 255, THRESH_OTSU);
		threshold(binaryImage, binaryImage, 0, 255, THRESH_BINARY_INV);
		//������̬ѧ������ϸ����Ե
		//reference http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/
		cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
		cv::Mat skel(binaryImage.size(), CV_8UC1, cv::Scalar(0));
		cv::Mat temp(binaryImage.size(), CV_8UC1);
		bool done;
		do
		{
			cv::morphologyEx(binaryImage, temp, cv::MORPH_OPEN, element);
			cv::bitwise_not(temp, temp);
			cv::bitwise_and(binaryImage, temp, temp);
			cv::bitwise_or(skel, temp, skel);
			cv::erode(binaryImage, binaryImage, element);
			double max;
			cv::minMaxLoc(binaryImage, 0, &max);
			done = (max == 0);
		} while (!done);

		vector<int> v_lines;
		if (direction == 0)
		{
			//���ݹ۲�ͳ��ÿһ�����صĸ���
			int ithresh = cvRound(skel.cols*thresh);
			vector<int> v_statics;
			for (size_t i = 0; i < skel.rows; i++)
			{
				int itmp = countNonZero(skel.row(i));
				v_statics.push_back(itmp);
			}
			for (size_t i = 0; i < v_statics.size() - 1; i++)
			{
				if ((v_statics[i] >= ithresh && v_statics[i + 1] < ithresh)) //|| (v_statics[i] < thresh && v_statics[i + 1] >= thresh)
				{
					v_lines.push_back(i);
				}
			}
			/*for (size_t i = 0; i < v_lines.size(); i++)
			{
				line(thumbnailImage, Point(0, v_lines[i]), Point(srcImage.cols - 1, v_lines[i]), Scalar(0, 0, 255), 1);
			}
			imshow("��ȡ������", thumbnailImage);*/
		}
		else {
			int ithresh = cvRound(skel.rows*thresh);
			vector<int> v_statics;
			for (size_t i = 0; i < skel.cols; i++)
			{
				int itmp = countNonZero(skel.col(i));
				v_statics.push_back(itmp);
			}
			for (size_t i = 0; i < v_statics.size() - 1; i++)
			{
				if ((v_statics[i] >= ithresh && v_statics[i + 1] < ithresh)) //|| (v_statics[i] < thresh && v_statics[i + 1] >= thresh)
				{
					v_lines.push_back(i);
				}
			}

			/*for (size_t i = 0; i < v_lines.size(); i++)
			 {
				 line(thumbnailImage, Point(v_lines[i],0), Point(v_lines[i],srcImage.rows - 1), Scalar(0, 0, 255), 1);
			 }
			 //imshow("��ȡ������", thumbnailImage);*/
		}

		if (v_lines.size() < 3)
		{
			cout << "ͼƬ�е�ֱ����ȡ����3��" << endl;
			return;
		}
		//ͳ�Ƹ���  �׵���һ��
		vector<int> vpt;
		for (size_t i = 1; i < v_lines.size() - 1; i++)
		{
			vpt.push_back((v_lines[i + 1] - v_lines[i])*ratio);
		}
		//����
		sort(vpt.begin(), vpt.end());
		//ѡ����λ��
		int count = vpt[vpt.size() / 2];
		cout << "pixels count:" << count << endl;
		pixelSize = 1. / count;
	}

	/*
	��ȡ���̸�ǵ㣬�������ڽ�������֮�����Ԫ������������ƽ��ֵ
	//srcImage  ����ͼ��
	//cellPixelCount  ���ͳ����Ԫ����
	*/
	void getCellPixelCount(const Mat& srcImage,int &cellPixelCount)
	{
		if (!srcImage.data)
		{
			cout << "data load failed" << endl;
			return;
		}
		Mat tmp = srcImage.clone();
		//��ȡ�������� ��ֹ����������
		Mat subImage = tmp(Rect(tmp.cols / 4, tmp.rows / 4, tmp.cols / 2, tmp.rows / 2));
		cv::Mat imageGray;
		cv::cvtColor(subImage, imageGray, CV_RGB2GRAY);
		Mat closeImage;
		Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
		morphologyEx(imageGray, closeImage, MORPH_CLOSE, element);
		Mat binarayImage;
		threshold(closeImage, binarayImage, 0, 255, THRESH_OTSU);
		assert(binarayImage.channels() == 1);
		//������е���Ԫ����
		int colNumber = binarayImage.cols;
		int rowNumber = binarayImage.rows;
		vector<int> v_pixel;
		for (size_t i = 0; i < rowNumber; i++)
		{
			uchar* data = binarayImage.ptr<uchar>(i);
			vector<int> v_count;
			for (size_t j = 0; j < colNumber; j++)
			{
				int ct = 0;
				while ((unsigned)data[j] == 255)
				{
					ct++;
					j++;
				}
				if (ct>0)
				{
					v_count.push_back(ct);
				}
			}
			sort(v_count.begin(), v_count.end());
			v_pixel.push_back(v_count[v_count.size() / 2]);
		}
		sort(v_pixel.begin(), v_pixel.end());
		cout << "ͳ�ƽ��" << endl;
		for (size_t i = 0; i < v_pixel.size(); i++)
		{
			printf("%2d ", v_pixel[i]);
		}
		cellPixelCount = v_pixel[v_pixel.size() / 2];
		

		//cv::Mat Extractcorner;
		//cv::vector<cv::Point2f> corners;    //���������� �нǵ�����
		//cv::Size board_size = cv::Size(4,4 );   //�궨��ÿ�У�ÿ�нǵ���
		//Extractcorner = image.clone();

		///*cv::Mat imageGray;
		//cv::cvtColor(image, imageGray, CV_RGB2GRAY);*/
		//bool patternfound = cv::findChessboardCorners(image, board_size, corners); //cv::CALIB_CB_FAST_CHECK + cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE
		//if (!patternfound)
		//{
		//	std::cout << "can not find chessboard corners!" << std::endl;
		//	exit(1);
		//}
		//else
		//{
		//	//�����ؾ�ȷ��
		//	cv::cornerSubPix(imageGray, corners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
		//}

		////�ǵ���ͼ����ʾ
		//for (int i = 0; i < corners.size(); i++)
		//{
		//	cv::circle(Extractcorner, corners[i], 5, cv::Scalar(255, 0, 255), 2);
		//}
		//cv::imshow("Extractcorner", Extractcorner);

		//cv::waitKey(0);
	}

}



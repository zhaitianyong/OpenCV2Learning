#pragma once
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>  
#include <opencv2\imgproc\imgproc.hpp>
#include <vector>
namespace atway
{
	using namespace std;
	using namespace cv;
	//ͶӰ�ָ��㷨
	void projectionSegmentation(const Mat& src, vector<int>& lines, int thresh, int direction = 0,bool isBoth=false);
	//��õ������̸����ظ���
	void getCellPixelCount(const Mat& srcImage, int& cellPixelCount);
	//������Ԫ�ߴ�
	void measure_ruler_center(const Mat& srcImage, double& pixelSize, double thresh=0.5, int direction = 0);
	void measure_ruler_edge(const Mat& srcImage, double& pixelSize, double thresh, int direction);
}
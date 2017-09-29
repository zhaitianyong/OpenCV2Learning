/************************************************************************
> File Name: 
> Author:atway
> Mail:atway#126.com(#=>@)
> Created Time: 2014年10月15日 星期三 12时00分33秒
************************************************************************/

//标定物体距离


#include "stdafx.h"
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>  
#include <opencv2\imgproc\imgproc.hpp>
#include <iostream>  
#include "improcess.h"
#define GRAY 128
#define WHITE 255
#define BLACK 0
using namespace std;
using namespace cv;

void cvHilditchThin2(cv::Mat& src, cv::Mat& dst)
{
	//http://cgm.cs.mcgill.ca/~godfried/teaching/projects97/azar/skeleton.html#algorithm
	//算法有问题，得不到想要的效果
	if (src.type() != CV_8UC1)
	{
		printf("只能处理二值或灰度图像\n");
		return;
	}
	//非原地操作时候，copy src到dst
	if (dst.data != src.data)
	{
		src.copyTo(dst);
	}

	int i, j;
	int width, height;
	//之所以减2，是方便处理8邻域，防止越界
	width = src.cols - 2;
	height = src.rows - 2;
	int step = src.step;
	int  p2, p3, p4, p5, p6, p7, p8, p9;
	uchar* img;
	bool ifEnd;
	int A1;
	cv::Mat tmpimg;
	while (1)
	{
		dst.copyTo(tmpimg);
		ifEnd = false;
		img = tmpimg.data + step;
		for (i = 2; i < height; i++)
		{
			img += step;
			for (j = 2; j<width; j++)
			{
				uchar* p = img + j;
				A1 = 0;
				if (p[0] > 0)
				{
					if (p[-step] == 0 && p[-step + 1]>0) //p2,p3 01模式
					{
						A1++;
					}
					if (p[-step + 1] == 0 && p[1]>0) //p3,p4 01模式
					{
						A1++;
					}
					if (p[1] == 0 && p[step + 1]>0) //p4,p5 01模式
					{
						A1++;
					}
					if (p[step + 1] == 0 && p[step]>0) //p5,p6 01模式
					{
						A1++;
					}
					if (p[step] == 0 && p[step - 1]>0) //p6,p7 01模式
					{
						A1++;
					}
					if (p[step - 1] == 0 && p[-1]>0) //p7,p8 01模式
					{
						A1++;
					}
					if (p[-1] == 0 && p[-step - 1]>0) //p8,p9 01模式
					{
						A1++;
					}
					if (p[-step - 1] == 0 && p[-step]>0) //p9,p2 01模式
					{
						A1++;
					}
					p2 = p[-step]>0 ? 1 : 0;
					p3 = p[-step + 1]>0 ? 1 : 0;
					p4 = p[1]>0 ? 1 : 0;
					p5 = p[step + 1]>0 ? 1 : 0;
					p6 = p[step]>0 ? 1 : 0;
					p7 = p[step - 1]>0 ? 1 : 0;
					p8 = p[-1]>0 ? 1 : 0;
					p9 = p[-step - 1]>0 ? 1 : 0;
					//计算AP2,AP4
					int A2, A4;
					A2 = 0;
					//if(p[-step]>0)
					{
						if (p[-2 * step] == 0 && p[-2 * step + 1]>0) A2++;
						if (p[-2 * step + 1] == 0 && p[-step + 1]>0) A2++;
						if (p[-step + 1] == 0 && p[1]>0) A2++;
						if (p[1] == 0 && p[0]>0) A2++;
						if (p[0] == 0 && p[-1]>0) A2++;
						if (p[-1] == 0 && p[-step - 1]>0) A2++;
						if (p[-step - 1] == 0 && p[-2 * step - 1]>0) A2++;
						if (p[-2 * step - 1] == 0 && p[-2 * step]>0) A2++;
					}


					A4 = 0;
					//if(p[1]>0)
					{
						if (p[-step + 1] == 0 && p[-step + 2]>0) A4++;
						if (p[-step + 2] == 0 && p[2]>0) A4++;
						if (p[2] == 0 && p[step + 2]>0) A4++;
						if (p[step + 2] == 0 && p[step + 1]>0) A4++;
						if (p[step + 1] == 0 && p[step]>0) A4++;
						if (p[step] == 0 && p[0]>0) A4++;
						if (p[0] == 0 && p[-step]>0) A4++;
						if (p[-step] == 0 && p[-step + 1]>0) A4++;
					}

					//printf("p2=%d p3=%d p4=%d p5=%d p6=%d p7=%d p8=%d p9=%d\n", p2, p3, p4, p5, p6,p7, p8, p9);
					//printf("A1=%d A2=%d A4=%d\n", A1, A2, A4);
					if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9)>1 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9)<7 && A1 == 1)
					{
						if (((p2 == 0 || p4 == 0 || p8 == 0) || A2 != 1) && ((p2 == 0 || p4 == 0 || p6 == 0) || A4 != 1))
						{
							dst.at<uchar>(i, j) = 0; //满足删除条件，设置当前像素为0
							ifEnd = true;
							//printf("\n");

							//PrintMat(dst);
						}
					}
				}
			}
		}
		//printf("\n");
		//PrintMat(dst);
		//PrintMat(dst);
		//已经没有可以细化的像素了，则退出迭代
		if (!ifEnd) break;
	}
}

int func_nc8(int *b)
//端点的连通性检测
{
	int n_odd[4] = { 1, 3, 5, 7 };  //四邻域
	int i, j, sum, d[10];

	for (i = 0; i <= 9; i++) {
		j = i;
		if (i == 9) j = 1;
		if (abs(*(b + j)) == 1)
		{
			d[i] = 1;
		}
		else
		{
			d[i] = 0;
		}
	}
	sum = 0;
	for (i = 0; i < 4; i++)
	{
		j = n_odd[i];
		sum = sum + d[j] - d[j] * d[j + 1] * d[j + 2];
	}
	return (sum);
}

void cvHilditchThin(cv::Mat& src, cv::Mat& dst)
{
	if (src.type() != CV_8UC1)
	{
		printf("只能处理二值或灰度图像\n");
		return;
	}
	//非原地操作时候，copy src到dst
	if (dst.data != src.data)
	{
		src.copyTo(dst);
	}

	//8邻域的偏移量
	int offset[9][2] = { { 0,0 },{ 1,0 },{ 1,-1 },{ 0,-1 },{ -1,-1 },
	{ -1,0 },{ -1,1 },{ 0,1 },{ 1,1 } };
	//四邻域的偏移量
	int n_odd[4] = { 1, 3, 5, 7 };
	int px, py;
	int b[9];                      //3*3格子的灰度信息
	int condition[6];              //1-6个条件是否满足
	int counter;                   //移去像素的数量
	int i, x, y, copy, sum;

	uchar* img;
	int width, height;
	width = dst.cols;
	height = dst.rows;
	img = dst.data;
	int step = dst.step;
	do
	{

		counter = 0;

		for (y = 0; y < height; y++)
		{

			for (x = 0; x < width; x++)
			{

				//前面标记为删除的像素，我们置其相应邻域值为-1
				for (i = 0; i < 9; i++)
				{
					b[i] = 0;
					px = x + offset[i][0];
					py = y + offset[i][1];
					if (px >= 0 && px < width &&    py >= 0 && py <height)
					{
						// printf("%d\n", img[py*step+px]);
						if (img[py*step + px] == WHITE)
						{
							b[i] = 1;
						}
						else if (img[py*step + px] == GRAY)
						{
							b[i] = -1;
						}
					}
				}
				for (i = 0; i < 6; i++)
				{
					condition[i] = 0;
				}

				//条件1，是前景点
				if (b[0] == 1) condition[0] = 1;

				//条件2，是边界点
				sum = 0;
				for (i = 0; i < 4; i++)
				{
					sum = sum + 1 - abs(b[n_odd[i]]);
				}
				if (sum >= 1) condition[1] = 1;

				//条件3， 端点不能删除
				sum = 0;
				for (i = 1; i <= 8; i++)
				{
					sum = sum + abs(b[i]);
				}
				if (sum >= 2) condition[2] = 1;

				//条件4， 孤立点不能删除
				sum = 0;
				for (i = 1; i <= 8; i++)
				{
					if (b[i] == 1) sum++;
				}
				if (sum >= 1) condition[3] = 1;

				//条件5， 连通性检测
				if (func_nc8(b) == 1) condition[4] = 1;

				//条件6，宽度为2的骨架只能删除1边
				sum = 0;
				for (i = 1; i <= 8; i++)
				{
					if (b[i] != -1)
					{
						sum++;
					}
					else
					{
						copy = b[i];
						b[i] = 0;
						if (func_nc8(b) == 1) sum++;
						b[i] = copy;
					}
				}
				if (sum == 8) condition[5] = 1;

				if (condition[0] && condition[1] && condition[2] && condition[3] && condition[4] && condition[5])
				{
					img[y*step + x] = GRAY; //可以删除，置位GRAY，GRAY是删除标记，但该信息对后面像素的判断有用
					counter++;
					//printf("----------------------------------------------\n");
					//PrintMat(dst);
				}
			}
		}

		if (counter != 0)
		{
			for (y = 0; y < height; y++)
			{
				for (x = 0; x < width; x++)
				{
					if (img[y*step + x] == GRAY)
						img[y*step + x] = BLACK;

				}
			}
		}

	} while (counter != 0);

}

void cvThin(cv::Mat& src, cv::Mat& dst, int intera)
{
	if (src.type() != CV_8UC1)
	{
		printf("只能处理二值或灰度图像\n");
		return;
	}
	//非原地操作时候，copy src到dst
	if (dst.data != src.data)
	{
		src.copyTo(dst);
	}

	int i, j, n;
	int width, height;
	width = src.cols - 1;
	//之所以减1，是方便处理8邻域，防止越界
	height = src.rows - 1;
	int step = src.step;
	int  p2, p3, p4, p5, p6, p7, p8, p9;
	uchar* img;
	bool ifEnd;
	int A1;
	cv::Mat tmpimg;
	//n表示迭代次数
	for (n = 0; n<intera; n++)
	{
		dst.copyTo(tmpimg);
		ifEnd = false;
		img = tmpimg.data;
		for (i = 1; i < height; i++)
		{
			img += step;
			for (j = 1; j<width; j++)
			{
				uchar* p = img + j;
				A1 = 0;
				if (p[0] > 0)
				{
					if (p[-step] == 0 && p[-step + 1]>0) //p2,p3 01模式
					{
						A1++;
					}
					if (p[-step + 1] == 0 && p[1]>0) //p3,p4 01模式
					{
						A1++;
					}
					if (p[1] == 0 && p[step + 1]>0) //p4,p5 01模式
					{
						A1++;
					}
					if (p[step + 1] == 0 && p[step]>0) //p5,p6 01模式
					{
						A1++;
					}
					if (p[step] == 0 && p[step - 1]>0) //p6,p7 01模式
					{
						A1++;
					}
					if (p[step - 1] == 0 && p[-1]>0) //p7,p8 01模式
					{
						A1++;
					}
					if (p[-1] == 0 && p[-step - 1]>0) //p8,p9 01模式
					{
						A1++;
					}
					if (p[-step - 1] == 0 && p[-step]>0) //p9,p2 01模式
					{
						A1++;
					}
					p2 = p[-step]>0 ? 1 : 0;
					p3 = p[-step + 1]>0 ? 1 : 0;
					p4 = p[1]>0 ? 1 : 0;
					p5 = p[step + 1]>0 ? 1 : 0;
					p6 = p[step]>0 ? 1 : 0;
					p7 = p[step - 1]>0 ? 1 : 0;
					p8 = p[-1]>0 ? 1 : 0;
					p9 = p[-step - 1]>0 ? 1 : 0;
					if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9)>1 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9)<7 && A1 == 1)
					{
						if ((p2 == 0 || p4 == 0 || p6 == 0) && (p4 == 0 || p6 == 0 || p8 == 0)) //p2*p4*p6=0 && p4*p6*p8==0
						{
							dst.at<uchar>(i, j) = 0; //满足删除条件，设置当前像素为0
							ifEnd = true;
						}
					}
				}
			}
		}

		dst.copyTo(tmpimg);
		img = tmpimg.data;
		for (i = 1; i < height; i++)
		{
			img += step;
			for (j = 1; j<width; j++)
			{
				A1 = 0;
				uchar* p = img + j;
				if (p[0] > 0)
				{
					if (p[-step] == 0 && p[-step + 1]>0) //p2,p3 01模式
					{
						A1++;
					}
					if (p[-step + 1] == 0 && p[1]>0) //p3,p4 01模式
					{
						A1++;
					}
					if (p[1] == 0 && p[step + 1]>0) //p4,p5 01模式
					{
						A1++;
					}
					if (p[step + 1] == 0 && p[step]>0) //p5,p6 01模式
					{
						A1++;
					}
					if (p[step] == 0 && p[step - 1]>0) //p6,p7 01模式
					{
						A1++;
					}
					if (p[step - 1] == 0 && p[-1]>0) //p7,p8 01模式
					{
						A1++;
					}
					if (p[-1] == 0 && p[-step - 1]>0) //p8,p9 01模式
					{
						A1++;
					}
					if (p[-step - 1] == 0 && p[-step]>0) //p9,p2 01模式
					{
						A1++;
					}
					p2 = p[-step]>0 ? 1 : 0;
					p3 = p[-step + 1]>0 ? 1 : 0;
					p4 = p[1]>0 ? 1 : 0;
					p5 = p[step + 1]>0 ? 1 : 0;
					p6 = p[step]>0 ? 1 : 0;
					p7 = p[step - 1]>0 ? 1 : 0;
					p8 = p[-1]>0 ? 1 : 0;
					p9 = p[-step - 1]>0 ? 1 : 0;
					if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9)>1 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9)<7 && A1 == 1)
					{
						if ((p2 == 0 || p4 == 0 || p8 == 0) && (p2 == 0 || p6 == 0 || p8 == 0)) //p2*p4*p8=0 && p2*p6*p8==0
						{
							dst.at<uchar>(i, j) = 0; //满足删除条件，设置当前像素为0
							ifEnd = true;
						}
					}
				}
			}
		}

		//如果两个子迭代已经没有可以细化的像素了，则退出迭代
		if (!ifEnd) break;
	}

}

void main_measure()
{
	Mat srcImage, grayImage;
	srcImage = imread("data\\尺度测量\\IMG_0002.JPG");
	if (!srcImage.data)
	{
		cout << "data load failed" << endl;
		return;
	}
	double pixelSize = 0.;
	//atway::measure_ruler_center(srcImage, pixelSize,0.3,0);
	atway::measure_ruler_edge(srcImage, pixelSize, 0.3, 0);
	double imageWidthMM = pixelSize*srcImage.size().width;
	double imageHeightMM = pixelSize*srcImage.size().height;
	cout << "each pixel size :" << pixelSize << endl;
	cout << "imageWidthMM :" << imageWidthMM << " mm" << endl;
	cout << "imageHeightMM :" << imageHeightMM << " mm" << endl;
	waitKey(0);
	system("pause");

}
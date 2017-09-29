/*
  Detects SIFT features in two images and finds matches between them.

  Copyright (C) 2006-2012  Rob Hess <rob@iqengines.com>

  @version 1.1.2-20100521
*/

#include "sift.h"
#include "imgfeatures.h"
#include "kdtree.h"
#include "utils.h"
#include "xform.h"

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

#include <stdio.h>


/* the maximum number of keypoint NN candidates to check during BBF search */
#define KDTREE_BBF_MAX_NN_CHKS 200

/* threshold on squared ratio of distances between NN and 2nd NN */
#define NN_SQ_DIST_RATIO_THR 0.49


int main_sift_test(int argc, char** argv)
{
	IplImage* img1, *img2, *stacked;
	struct feature* feat1, *feat2, *feat;
	struct feature** nbrs;
	struct kd_node* kd_root;
	CvPoint pt1, pt2;
	double d0, d1;
	int n1, n2, k, i, m = 0;

	if (argc != 3)
		fatal_error("usage: %s <img1> <img2>", argv[0]);
	//加载图片1
	img1 = cvLoadImage(argv[1], 1);
	if (!img1)
		fatal_error("unable to load image from %s", argv[1]);
	//加载图片2
	img2 = cvLoadImage(argv[2], 1);
	if (!img2)
		fatal_error("unable to load image from %s", argv[2]);
	//把图片1和图片2堆叠在一起
	stacked = stack_imgs(img1, img2);

	fprintf(stderr, "Finding features in %s...\n", argv[1]);
	n1 = sift_features(img1, &feat1);
	fprintf(stderr, "Finding features in %s...\n", argv[2]);
	n2 = sift_features(img2, &feat2);
	fprintf(stderr, "Building kd tree...\n");
	kd_root = kdtree_build(feat2, n2);  //对img2的特征向量进行构建kdtree 方便查找
	for (i = 0; i < n1; i++)
	{
		feat = feat1 + i;
		//快速最近领域查找 这里k=2 查找两个最近的点
		k = kdtree_bbf_knn(kd_root, feat, 2, &nbrs, KDTREE_BBF_MAX_NN_CHKS);
		if (k == 2)
		{
			d0 = descr_dist_sq(feat, nbrs[0]); //计算最近点距离
			d1 = descr_dist_sq(feat, nbrs[1]); //计算次近点距离
			if (d0 < d1 * NN_SQ_DIST_RATIO_THR)
			{
				//pt1 和 pt2 就是一对匹配的点
				pt1 = cvPoint(cvRound(feat->x), cvRound(feat->y));
				pt2 = cvPoint(cvRound(nbrs[0]->x), cvRound(nbrs[0]->y));
				pt2.y += img1->height;
				cvLine(stacked, pt1, pt2, CV_RGB(255, 0, 255), 1, 8, 0);
				m++;
				feat1[i].fwd_match = nbrs[0];
			}
		}
		free(nbrs);
	}
	fprintf(stderr, "Found %d total matches\n", m);
	display_big_img(stacked, "Matches");
	cvWaitKey(0);
	int display = 1;
	if (display)
	{
		draw_features(img1, feat1, n1);
		display_big_img(img1, argv[1]);
		cvWaitKey(0);
	}
	if (display)
	{
		draw_features(img2, feat2, n2);
		display_big_img(img2, argv[2]);
		cvWaitKey(0);
	}
	/*
	   UNCOMMENT BELOW TO SEE HOW RANSAC FUNCTION WORKS

	   Note that this line above:

	   feat1[i].fwd_match = nbrs[0];

	   is important for the RANSAC function to work.
	*/

	{
		struct feature** in_features;
		int  n = 0;
		CvMat* H;
		IplImage* xformed;
		H = ransac_xform(feat1, n1, FEATURE_FWD_MATCH, lsq_homog, 4, 0.01,
			homog_xfer_err, 3.0, &in_features, &n);
		fprintf(stderr, "Found %d total matches\n", n);
		if (H)
		{
			xformed = cvCreateImage(cvGetSize(img2), IPL_DEPTH_8U, 3);
			cvWarpPerspective(img1, xformed, H,
				CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS,
				cvScalarAll(0));
			cvNamedWindow("Xformed", 1);
			cvShowImage("Xformed", xformed);
			cvWaitKey(0);
			cvReleaseImage(&xformed);
			cvReleaseMat(&H);
		}

		


	}
	cvReleaseImage(&stacked);
	cvReleaseImage(&img1);
	cvReleaseImage(&img2);
	kdtree_release(kd_root);
	free(feat1);
	free(feat2);
	return 0;
}

homography.xml 为输出的变换矩阵
待变换的图是姚飞在办公室拍摄的
模板为小王提供的

我选择者了四对点进行图像变换：
basePoints为 模板上的点
srcPoints为  待变换图上的点

第一对
        basePoints.push_back(Point2f(163, 3346));
	srcPoints.push_back(Point2f(147, 3052));
第二对
	basePoints.push_back(Point2f(5019, 3346));
	srcPoints.push_back(Point2f(5810, 3045));
第三对
	basePoints.push_back(Point2f(163, 109));
	srcPoints.push_back(Point2f(801, 548));
第四对
	basePoints.push_back(Point2f(5018, 109));
	srcPoints.push_back(Point2f(5007, 633));
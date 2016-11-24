#include "psnr.h"

using namespace cv;

class PSNR
{
	double mse(Mat img_I, Mat img_K, int height, int width){
		double mse = 0;
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				mse += (img_I.at<double>(i, j) - img_K.at<double>(i, j)) * (img_I.at<double>(i, j) - img_K.at<double>(i, j));
		mse /= height * width;
		return mse;
	}

	double psnr(Mat img_I, Mat img_K){
		return (10 * log10((255 * 255) / mse(img_I, img_K, img_I.rows, img_K.cols)));
	}
};
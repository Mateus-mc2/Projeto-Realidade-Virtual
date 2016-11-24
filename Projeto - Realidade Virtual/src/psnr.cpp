#include "psnr.h"

using namespace cv;

class PSNR
{
	double mse(Mat img_I, Mat img_K, int height, int width){
		double mse = 0;
		double I, K, v;
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				I = img_I.at<double>(i, j);
				K = img_K.at<double>(i, j);
				v = I - K;
				mse += (v*v);
			}
		}
		mse /= ((double) (height * width));
		return mse;
	}

	double psnr(Mat img_I, Mat img_K){
		//psnr = 20*log(255) - 10*log(mse)
		return (48.1308 - 10*log10(mse(img_I, img_K, img_I.rows, img_K.cols)));
	}
};
#include <opencv/cv.h>
#include <opencv/highgui.h>

using namespace cv;

namespace util
{
	class PSNR {
	private:
		double mse(Mat img_I,
				   Mat img_K,
				   int height,
				   int weight) {}

	public:
		double psnr(Mat img_I, 
				  Mat img_K) {}
	};
}
#include <iostream>

#include <Eigen/Dense>
#include <opencv2/core.hpp>

typedef cv::Matx<int, 3, 3> CVMatrix3i;
using Eigen::Matrix3d;

int main() {
	Matrix3d eigen_identity = Matrix3d::Identity();
	std::cout << eigen_identity << std::endl;

	CVMatrix3i cv_identity = CVMatrix3i::eye();
	std::cout << cv_identity << std::endl;
	system("pause");

	return 0;
}
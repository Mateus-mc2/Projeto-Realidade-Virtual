#ifndef CAMERA_H_
#define CAMERA_H_

#include <Eigen/Dense>

namespace util {

struct Camera {
  Camera() {}
  Camera(const Eigen::Vector3d &eye, const Eigen::Vector2d &bottom, const Eigen::Vector2d &top,
         double width, double height)
      : eye_(eye),
        bottom_(bottom),
        top_(top),
        width_(width),
        height_(height) {}

  ~Camera() {}

  Eigen::Vector3d eye_;
  Eigen::Vector2d bottom_;
  Eigen::Vector2d top_;
  double width_;
  double height_;
};

}  // namespace util

#endif  // CAMERA_H_

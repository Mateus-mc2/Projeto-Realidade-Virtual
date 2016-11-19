#ifndef CAMERA_H_
#define CAMERA_H_

#include <Eigen/Dense>

namespace util {

struct Camera {
 public:
  Camera() {}
  Camera(const Eigen::Vector3f &eye, const Eigen::Vector2f &bottom, const Eigen::Vector2f &top,
         int width, int height)
      : eye(eye),
        bottom(bottom),
        top(top),
        width(width),
        height(height) {}

  ~Camera() {}

  Eigen::Vector3f eye;
  Eigen::Vector2f bottom;
  Eigen::Vector2f top;
  int width;
  int height;
};

}  // namespace util

#endif  // CAMERA_H_

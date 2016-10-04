#ifndef RAY_H_
#define RAY_H_

#include <Eigen\Dense>
#include <stack>

namespace util {

class RenderableObject;

struct Ray {
  Ray(const Eigen::Vector3d &origin, const Eigen::Vector3d &direction, int depth)
      : origin(origin),
        direction(direction),
        ambient_objs(std::stack<RenderableObject*>()),
        depth(depth) {}
  Ray(const Eigen::Vector3d &origin, const Eigen::Vector3d &direction,
      const std::stack<RenderableObject*> &objs, int depth)
      : origin(origin),
        direction(direction),
        ambient_objs(objs),
        depth(depth) {}
  ~Ray() {}

  Eigen::Vector3d origin;
  Eigen::Vector3d direction;
  std::stack<RenderableObject*> ambient_objs;
  int depth;
};

}  // namespace util

#endif  // RAY_H_
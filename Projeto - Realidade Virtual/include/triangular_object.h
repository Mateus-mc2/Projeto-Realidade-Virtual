#ifndef TRIANGULAR_OBJECT_H_
#define TRIANGULAR_OBJECT_H_

#include <Eigen/Dense>

#include <vector>

#include "math_lib.h"
#include "renderable_object.h"

namespace util {

class TriangularObject : public RenderableObject {
 public:
  TriangularObject(const Material &material,
                   bool is_emissive,
                   const std::vector<Eigen::Vector3d> &vertices,
                   const std::vector<Eigen::Vector3i> &faces);
  ~TriangularObject() {}

  double GetIntersectionParameter(const Ray &ray, Eigen::Vector3d *normal) const;
  const std::vector<Eigen::Vector3d> kVertices;
  const std::vector<Eigen::Vector3i> kFaces;

 private:
  bool IsInnerPoint(const Eigen::Vector3d &barycentric_coordinates) const;

  std::vector<Eigen::Vector4d> planes_coeffs_;
  std::vector<Eigen::Matrix3d> linear_systems_;  // Inverse system matrices.
};

}  // namespace util

#endif  // TRIANGULAR_OBJECT_H_

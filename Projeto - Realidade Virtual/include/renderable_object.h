#ifndef RENDERABLE_OBJECT_H_
#define RENDERABLE_OBJECT_H_

#include <Eigen/Dense>

#include "material.h"
#include "ray.h"

namespace util {

class RenderableObject {
 public:
  RenderableObject(const Material &material, bool is_emissive)
      :  kMaterial(material),
          kEmissive(is_emissive),
          kEps(1.0e-03) {}
  ~RenderableObject() {}

  virtual double GetIntersectionParameter(const Ray &ray, Eigen::Vector3d &normal) const = 0;

  // Accessors
  Material material() const { return this->kMaterial; }
  bool emissive() const { return this->kEmissive; }

 protected:
  const double kEps;
  const Material kMaterial;
  const bool kEmissive;
};

}  // namespace util

#endif  // RENDERABLE_OBJECT_H_

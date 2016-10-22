#ifndef QUADRIC_H_
#define QUADRIC_H_

#include <Eigen\Dense>

#include <string>

#include "math_lib.h"
#include "renderable_object.h"
#include "util_exception.h"

namespace util {

class InvalidCoefficientsVectorException : public UtilException {
 public:
  InvalidCoefficientsVectorException(const std::string &error_msg) : UtilException(error_msg) {}
};

class Quadric : public RenderableObject {
 public:
  // See http://mathworld.wolfram.com/QuadraticSurface.html to understand this notation.
  enum Index {kCoeffA, kCoeffB, kCoeffC,
              kCoeffF, kCoeffG, kCoeffH,
              kCoeffP, kCoeffQ, kCoeffR,
              kCoeffD};

  Quadric(const Eigen::VectorXd &coefficients, const Material &material, bool is_emissive);
  Quadric(double a, double b, double c, double f, double g, double h, double p, double q,
          double r, double d, const Material &material, bool is_emissive);
  ~Quadric() {}

  double GetIntersectionParameter(const Ray &ray, Eigen::Vector3d *normal) const;

  // Accessors.
  Eigen::VectorXd coefficients() const { return this->coefficients_; }

 private:
  Eigen::VectorXd coefficients_;
};

}  // namespace util

#endif  // QUADRIC_H_

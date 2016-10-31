#include "quadric.h"

using Eigen::Vector3d;
using Eigen::VectorXd;

namespace util {

Quadric::Quadric(const VectorXd &coefficients, const Material &material, bool is_emissive)
      :  RenderableObject(material, is_emissive),
         coefficients_(coefficients) {
  if (coefficients.size() != 10) {
    throw InvalidCoefficientsVectorException("Coefficients' vector doesn't have a valid size.");
  }
}

Quadric::Quadric(double a, double b, double c, double f, double g, double h, double p, double q,
                 double r, double d, const Material &material, bool is_emissive) 
                 :  RenderableObject(material, is_emissive) {
  this->coefficients_.resize(10);
  this->coefficients_(this->kCoeffA) = a;
  this->coefficients_(this->kCoeffB) = b;
  this->coefficients_(this->kCoeffC) = c;
  this->coefficients_(this->kCoeffF) = f;
  this->coefficients_(this->kCoeffG) = g;
  this->coefficients_(this->kCoeffH) = h;
  this->coefficients_(this->kCoeffP) = p;
  this->coefficients_(this->kCoeffQ) = q;
  this->coefficients_(this->kCoeffR) = r;
  this->coefficients_(this->kCoeffD) = d;
}

double Quadric::GetIntersectionParameter(const Ray &ray, Vector3d *normal) const {
  // Coefficients.
  double a = this->coefficients_(this->kCoeffA);
  double b = this->coefficients_(this->kCoeffB);
  double c = this->coefficients_(this->kCoeffC);

  double f = this->coefficients_(this->kCoeffF);
  double g = this->coefficients_(this->kCoeffG);
  double h = this->coefficients_(this->kCoeffH);

  double p = this->coefficients_(this->kCoeffP);
  double q = this->coefficients_(this->kCoeffQ);
  double r = this->coefficients_(this->kCoeffR);
  double d = this->coefficients_(this->kCoeffD);

  // Ray parameters (origin and direction).
  double x_0 = ray.origin(0);
  double y_0 = ray.origin(1);
  double z_0 = ray.origin(2);

  double dx = ray.direction(0);
  double dy = ray.direction(1);
  double dz = ray.direction(2);
  
  double t;  // Parameter to return.

  // Equation coefficients (degree 2: kA*t^2 + kB*t + kC = 0).
  const double kA = a*dx*dx + b*dy*dy + c*dz*dz + 2*(f*dy*dz + g*dx*dz + h*dx*dy);
  const double kB = 2*(a*x_0*dx + b*y_0*dy + c*z_0*dz + f*(dz*y_0 + dy*z_0) + g*(dz*x_0 + dx*z_0)
                    + h*(dy*x_0 + dx*y_0) + p*dx + q*dy+ r*dz);
  const double kC = a*x_0*x_0 + b*y_0*y_0 + c*z_0*z_0 + d + 2*(f*y_0*z_0 + g*x_0*z_0 + h*x_0*y_0
                    + p*x_0 + q*y_0 + r*z_0);

  if (math::IsAlmostEqual(kA, 0.0, this->kEps)) {
    // The equation has degree 1.
    if (math::IsAlmostEqual(kB, 0.0, this->kEps)) {
      // The equation has degree 0, thus it's degenerate (it has infinite - or even zero - roots).
      return -1.0;
    }

    t = (-kC) / kB;
  } else {
    double discriminant = kB*kB - 4*kA*kC;

    if (discriminant < 0.0) {
      // No real roots.
      return -1.0;
    } else if (math::IsAlmostEqual(discriminant, 0.0, this->kEps)) {
      t = (-kB) / (2*kA);
    } else {
      double sqrt_delta = std::sqrt(discriminant);
      // Gets the nearest point in front of the ray center.
      t = (-kB - sqrt_delta) / (2*kA);

      if (t < 0.0 || math::IsAlmostEqual(t, 0.0, this->kEps)) {
        // It is behind/coincident with the ray center.
        t = (-kB + sqrt_delta) / (2*kA);
      }
    }
  }

  // Get normal from this point - must get the gradient from the implicit equation.
  double x = x_0 + t*dx;
  double y = y_0 + t*dy;
  double z = z_0 + t*dz;

  (*normal)(0) = 2*(a*x + g*z + h*y + p);
  (*normal)(1) = 2*(b*y + f*z + h*x + q);
  (*normal)(2) = 2*(c*z + f*y + g*x + r);

  if (!(math::IsAlmostEqual((*normal)(0), 0.0, this->kEps) &&
        math::IsAlmostEqual((*normal)(1), 0.0, this->kEps) &&
        math::IsAlmostEqual((*normal)(2), 0.0, this->kEps))) {
    (*normal) = (*normal) / normal->norm();
  }

  if (t < this->kEps) {  // If it's negative or almost zero.
    return -1.0;
  } else {
    return t;
  }
}

}  // namespace util

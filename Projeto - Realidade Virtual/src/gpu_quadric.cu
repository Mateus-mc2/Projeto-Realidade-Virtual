#include "gpu_quadric.h"

#include <thrust/complex.h>

#include "math_lib.h"
#include "quadric_coefficients.h"

namespace gpu {

GPUQuadric::GPUQuadric(float a, float b, float c, float f, float g, float h, float p, float q,
                       float r, float d, const GPUMaterial &material)
    : GPURenderableObject(material) {
  this->coefficients_[project::kCoeffB] = b;
  this->coefficients_[project::kCoeffA] = a;
  this->coefficients_[project::kCoeffC] = c;
  this->coefficients_[project::kCoeffF] = f;
  this->coefficients_[project::kCoeffG] = g;
  this->coefficients_[project::kCoeffH] = h;
  this->coefficients_[project::kCoeffP] = p;
  this->coefficients_[project::kCoeffQ] = q;
  this->coefficients_[project::kCoeffR] = r;
  this->coefficients_[project::kCoeffD] = d;
}

__host__ __device__ float GPUQuadric::GetIntersectionParameter(const GPURay &ray, float3 *normal) const {
  // Coefficients.
  float a = this->coefficients_[project::kCoeffA];
  float b = this->coefficients_[project::kCoeffB];
  float c = this->coefficients_[project::kCoeffC];

  float f = this->coefficients_[project::kCoeffF];
  float g = this->coefficients_[project::kCoeffG];
  float h = this->coefficients_[project::kCoeffH];

  float p = this->coefficients_[project::kCoeffP];
  float q = this->coefficients_[project::kCoeffQ];
  float r = this->coefficients_[project::kCoeffR];
  float d = this->coefficients_[project::kCoeffD];

  // Ray parameters (origin and direction).
  float x_0 = ray.origin.x;
  float y_0 = ray.origin.y;
  float z_0 = ray.origin.z;

  float dx = ray.direction.x;
  float dy = ray.direction.y;
  float dz = ray.direction.z;

  float t;  // Parameter to return.

  // Equation coefficients (degree 2: A*t^2 + B*t + C = 0).
  const float A = a*dx*dx + b*dy*dy + c*dz*dz + 2 * (f*dy*dz + g*dx*dz + h*dx*dy);
  const float B = 2 * (a*x_0*dx + b*y_0*dy + c*z_0*dz + f*(dz*y_0 + dy*z_0) + g*(dz*x_0 + dx*z_0)
                  + h*(dy*x_0 + dx*y_0) + p*dx + q*dy + r*dz);
  const float C = a*x_0*x_0 + b*y_0*y_0 + c*z_0*z_0 + d + 2 * (f*y_0*z_0 + g*x_0*z_0 + h*x_0*y_0
                  + p*x_0 + q*y_0 + r*z_0);

  if (math::IsAlmostEqual(A, 0.0f, this->kEps)) {
    // The equation has degree 1.
    if (math::IsAlmostEqual(B, 0.0f, this->kEps)) {
      // The equation has degree 0, thus it's degenerate (it has infinite - or even zero - roots).
      return -1.0;
    }

    t = (-C) / B;
  } else {
    float discriminant = B*B - 4 * A*C;

    if (discriminant < 0.0) {
      // No real roots.
      return -1.0;
    } else if (math::IsAlmostEqual(discriminant, 0.0f, this->kEps)) {
      t = (-B) / (2 * A);
    } else {
      float sqrt_delta = thrust::sqrt(thrust::complex<float>(discriminant)).real();
      // Gets the nearest point in front of the ray center.
      t = (-B - sqrt_delta) / (2 * A);

      if (t < 0.0 || math::IsAlmostEqual(t, 0.0f, this->kEps)) {
        // It is behind/coincident with the ray center.
        t = (-B + sqrt_delta) / (2 * A);
      }
    }
  }

  // Get normal from this point - must get the gradient from the implicit equation.
  float x = x_0 + t*dx;
  float y = y_0 + t*dy;
  float z = z_0 + t*dz;

  normal->x = 2 * (a*x + g*z + h*y + p);
  normal->y = 2 * (b*y + f*z + h*x + q);
  normal->z = 2 * (c*z + f*y + g*x + r);

  if (!(math::IsAlmostEqual(normal->x, 0.0f, this->kEps) &&
        math::IsAlmostEqual(normal->y, 0.0f, this->kEps) &&
        math::IsAlmostEqual(normal->z, 0.0f, this->kEps))) {
    math::Normalize(normal);
  }

  if (t < this->kEps) {  // If it's negative or almost zero.
    return -1.0;
  } else {
    return t;
  }
}

}  // namespace gpu

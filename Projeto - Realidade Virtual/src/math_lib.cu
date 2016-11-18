#include "math_lib.h"

namespace math {

__host__ bool IsAlmostEqual(double x, double y, double eps) {
  assert(eps < 1.0);
  return std::abs(x - y) <= std::abs(x)*eps;
}

__device__ bool IsAlmostEqual(float x, float y, float eps) {
  return thrust::abs(thrust::complex<float>(x - y)) <= thrust::abs(thrust::complex<float>(x))*eps;
}
__device__ void Normalize(float3 *triple) {
  float squared_norm = triple->x * triple->x + triple->y * triple->y + triple->z * triple->z;
  float norm = thrust::sqrt(thrust::complex<float>(squared_norm)).real();

  triple->x /= norm;
  triple->y /= norm;
  triple->z /= norm;
}

}  // namespace math

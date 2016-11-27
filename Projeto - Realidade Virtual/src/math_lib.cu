#include "math_lib.h"

namespace math {

__host__ __device__ bool IsAlmostEqual(double x, double y, double eps) {
  return abs(x - y) <= abs(x) * eps;
}

__host__ __device__ bool IsAlmostEqual(float x, float y, float eps) {
  return abs(x - y) <= abs(x) * eps;
}

__host__ __device__ void Normalize(float3 *triple) {
  float squared_norm = InnerProduct(*triple, *triple);
  float norm = sqrt(squared_norm);

  triple->x /= norm;
  triple->y /= norm;
  triple->z /= norm;
}

__host__ __device__ float InnerProduct(const float3 &u, const float3 &v) {
  return u.x * v.x + u.y * v.y + u.z * v.z;
}

__host__ __device__ float3 DirectProduct3(const float3 &u, const float3 &v) {
  return make_float3(u.x * v.x, u.y * v.y, u.z * v.z);
}

__host__ __device__ float3 Cross(const float3 &u, const float3 &v) {
  return make_float3(u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.y, u.x * v.y - u.y * v.x);
}

__host__ __device__ void Clamp3(float low, float high, float3 *triple) {
  triple->x = triple->x < low ? low : (triple->x < high ? high : triple->x);
  triple->y = triple->y < low ? low : (triple->y < high ? high : triple->y);
  triple->z = triple->z < low ? low : (triple->z < high ? high : triple->z);
}
}  // namespace math

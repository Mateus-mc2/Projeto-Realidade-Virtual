#ifndef MATH_LIB_H_
#define MATH_LIB_H_

#include <cuda_runtime.h>

namespace math {

__host__ __device__ bool IsAlmostEqual(double x, double y, double eps);
__host__ __device__ bool IsAlmostEqual(float x, float y, float eps);
__host__ __device__ void Normalize(float3 *triple);
__host__ __device__ float InnerProduct(const float3 &u, const float3 &v);
__host__ __device__ float3 DirectProduct3(const float3 &u, const float3 &v);
__host__ __device__ float3 Cross(const float3 &u, const float3 &v);
__host__ __device__ void Clamp3(float low, float high, float3 *triple);

}  // namespace math

#endif  // MATH_LIB_H_

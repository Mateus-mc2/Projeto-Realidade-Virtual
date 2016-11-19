#ifndef MATH_LIB_H_
#define MATH_LIB_H_

#include <cassert>
#include <cmath>

#include <cuda_runtime.h>
#include <thrust/complex.h>

namespace math {

__host__ __device__ bool IsAlmostEqual(double x, double y, double eps);
__host__ __device__ bool IsAlmostEqual(float x, float y, float eps);
__host__ __device__ void Normalize(float3 *triple);

}  // namespace math

#endif  // MATH_LIB_H_

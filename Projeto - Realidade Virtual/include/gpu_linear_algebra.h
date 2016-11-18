#ifndef GPU_LINEAR_ALGEBRA_H_
#define GPU_LINEAR_ALGEBRA_H_

#include "gpu_matrix.h"

namespace gpu {

__device__ void LUPDecomposition(GPUMatrix *L, GPUMatrix *U, GPUMatrix *P);
__device__ void ApplyForwardSubstitution(const GPUMatrix &L, const GPUMatrix &b, GPUMatrix *x);
__device__ void ApplyBackSubstitution(const GPUMatrix &U, const GPUMatrix &b, GPUMatrix *x);

}  // namespace gpu

#endif  // GPU_LINEAR_ALGEBRA_H_

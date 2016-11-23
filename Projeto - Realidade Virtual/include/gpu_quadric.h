#ifndef GPU_QUADRIC_H_
#define GPU_QUADRIC_H_

#include <cuda_runtime.h>

#include "gpu_material.h"
#include "gpu_ray.h"
#include "gpu_renderable_object.h"

namespace gpu {

class GPUQuadric : public GPURenderableObject {
 public:
  GPUQuadric() {}
  GPUQuadric(float a, float b, float c, float f, float g, float h, float p,
             float q, float r, float d, const GPUMaterial &material);
  ~GPUQuadric() {}

  __host__ __device__ float GetIntersectionParameter(const GPURay &ray, float3 *normal) const;

  // Accessors.
  __host__ __device__ const float* coefficients() const { return this->coefficients_; }

 private:
  float coefficients_[10];
};

}  // namespace gpu

#endif  // GPU_QUADRIC_H_

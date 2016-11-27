#ifndef GPU_QUADRIC_H_
#define GPU_QUADRIC_H_

#include <cuda_runtime.h>

#include "gpu_array.h"
#include "gpu_material.h"
#include "gpu_ray.h"

namespace gpu {

class GPUQuadric {
 public:
  typedef GPUArray<float, 10> Coefficients;

  GPUQuadric() : kEps(1.0e-3f) {}
  GPUQuadric(const GPUQuadric &quadric);
  GPUQuadric(float a, float b, float c, float f, float g, float h, float p, float q, float r,
             float d, const GPUMaterial &material);
  ~GPUQuadric() {}

  GPUQuadric& operator=(const GPUQuadric &quadric);

  __host__ __device__ float GetIntersectionParameter(const GPURay &ray, float3 *normal) const;

  // Accessors.
  __host__ __device__ const GPUMaterial& material() const { return this->material_; }
  //const Coefficients& coefficients() const { return this->coefficients_; }
  __host__ __device__ const float* coefficients() const { return this->coefficients_; }

 private:
  const float kEps;
  GPUMaterial material_;
  //Coefficients coefficients_;
  float coefficients_[10];
};

}  // namespace gpu

#endif  // GPU_QUADRIC_H_

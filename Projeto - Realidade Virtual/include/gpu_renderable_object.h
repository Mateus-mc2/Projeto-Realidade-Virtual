#ifndef GPU_RENDERABLE_OBJECT_H_
#define GPU_RENDERABLE_OBJECT_H_

#include <cuda_runtime.h>

#include "gpu_material.h"
#include "gpu_ray.h"

namespace gpu {

struct GPURay;

class GPURenderableObject {
 public:
  __host__ __device__ GPURenderableObject() : kEps(1.0e-03f) {}
  __host__ __device__ explicit GPURenderableObject(const GPUMaterial &material)
      : material_(material),
        kEps(1.0e-03f) {}
  __host__ __device__ ~GPURenderableObject() {}

  __host__ __device__ virtual float GetIntersectionParameter(const GPURay &ray, float3 *normal)
      const = 0;

  // Accessors.
  __host__ __device__ const GPUMaterial& material() const { return this->material_; }

 protected:
  const float kEps;
  GPUMaterial material_;
};

}  // namespace gpu

#endif  // GPU_RENDERABLE_OBJECT_H_

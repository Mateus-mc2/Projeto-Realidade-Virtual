#ifndef GPU_TRIANGULAR_OBJECT_H_
#define GPU_TRIANGULAR_OBJECT_H_

#include <cuda_runtime.h>

#include "gpu_material.h"
#include "gpu_matrix.h"
#include "gpu_ray.h"
#include "gpu_vector.h"
#include "gpu_renderable_object.h"

namespace gpu {

class GPUTriangularObject {
 public:
  GPUTriangularObject()
      : kEps(1.0e-3f) {}
  GPUTriangularObject(const GPUTriangularObject &obj)
      : kEps(1.0e-3f),
        material_(obj.material()),
        planes_coeffs_(obj.planes_coeffs()),
        linear_systems_(obj.linear_systems()) {}
  GPUTriangularObject(const GPUMaterial &material, const GPUVector<float3> &vertices,
                      const GPUVector<int3> &faces);
  ~GPUTriangularObject() {}

  GPUTriangularObject& operator=(const GPUTriangularObject &obj);

  __host__ __device__ float GetIntersectionParameter(const GPURay &ray, float3 *normal) const;

  // Accessors.
  __host__ __device__ const GPUMaterial& material() const { return this->material_; }
  __host__ __device__ const GPUVector<float4>& planes_coeffs() const {
    return this->planes_coeffs_;
  }
  __host__ __device__ const GPUVector<GPUMatrix>& linear_systems() const {
    return this->linear_systems_;
  }

 private:
  const float kEps;
  GPUMaterial material_;
  GPUVector<float4> planes_coeffs_;      // Plane coefficients where each face belongs.
  GPUVector<GPUMatrix> linear_systems_;  // Upper triangular system matrices.
};

}  // namespace gpu


#endif  // GPU_TRIANGULAR_OBJECT_H_

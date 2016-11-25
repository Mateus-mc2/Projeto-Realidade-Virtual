#ifndef GPU_TRIANGULAR_OBJECT_H_
#define GPU_TRIANGULAR_OBJECT_H_

#include <cuda_runtime.h>

#include "gpu_material.h"
#include "gpu_matrix.h"
#include "gpu_ray.h"
#include "gpu_renderable_object.h"

namespace gpu {

class GPUTriangularObject {
 public:
  __host__ __device__ GPUTriangularObject()
      : kEps(1.0e-3f),
        planes_coeffs_(nullptr),
        linear_systems_(nullptr),
        num_faces_(0) {}
  __host__ __device__ GPUTriangularObject(const GPUTriangularObject &object);
  __host__ __device__ GPUTriangularObject(const GPUMaterial &material, const float3 *vertices,
                                          const int3 *faces, int num_faces);
  __host__ __device__ ~GPUTriangularObject() {
    if (!this->planes_coeffs_) delete[] this->planes_coeffs_;
    if (!this->linear_systems_) delete[] this->linear_systems_;
  }

  __host__ __device__ GPUTriangularObject& operator=(const GPUTriangularObject &obj);

  __host__ __device__ float GetIntersectionParameter(const GPURay &ray, float3 *normal) const;

  // Accessors.
  __host__ __device__ const GPUMaterial& material() const { return this->material_; }
  __host__ __device__ const float4* planes_coeffs() const { return this->planes_coeffs_; }
  __host__ __device__ const GPUMatrix* linear_systems() const { return this->linear_systems_; }
  __host__ __device__ int num_faces() const { return this->num_faces_; }

 private:
  const float kEps;
  GPUMaterial material_;
  float4* planes_coeffs_;      // Plane coefficients where each face belongs.
  GPUMatrix* linear_systems_;  // Upper triangular system matrices.
  int num_faces_;
};

}  // namespace gpu


#endif  // GPU_TRIANGULAR_OBJECT_H_

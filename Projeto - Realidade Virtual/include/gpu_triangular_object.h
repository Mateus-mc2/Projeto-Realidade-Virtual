#ifndef GPU_TRIANGULAR_OBJECT_H_
#define GPU_TRIANGULAR_OBJECT_H_

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "gpu_material.h"
#include "gpu_matrix.h"
#include "gpu_ray.h"
#include "gpu_renderable_object.h"

namespace gpu {

class GPUTriangularObject : public GPURenderableObject {
 public:
  __host__ __device__ GPUTriangularObject(const GPUMaterial &material,
                                          const thrust::device_vector<float3> &vertices, 
                                          const thrust::device_vector<float4> &faces);
  __device__ ~GPUTriangularObject();

  __device__ float GetIntersectionParameter(const GPURay &ray, float3 *normal) const;

 private:
  bool IsInnerPoint(const float3 &barycentric_coordinates) const;

  float4* planes_coeffs_;   // Plane coefficients where each face belongs.
  GPUMatrix3f* linear_systems_;  // Upper triangular system matrices.
  int num_faces;
};

}  // namespace gpu


#endif  // GPU_TRIANGULAR_OBJECT_H_

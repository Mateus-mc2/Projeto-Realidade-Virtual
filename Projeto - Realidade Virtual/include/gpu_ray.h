#ifndef GPU_RAY_H_
#define GPU_RAY_H_

#include <cuda_runtime.h>

#include "gpu_renderable_object.h"
#include "gpu_stack.h"

namespace gpu {

class GPURenderableObject;

// TODO(Mateus): implement a stack on device (thrust::device_vector cannot be used on device).
struct GPURay {
 public:
  __device__ GPURay(const float3 &origin, const float3 &direction)
      : origin(origin),
        direction(direction),
        depth(1) {}
  __device__ GPURay(const float3 &origin, const float3 &direction, int depth)
      : origin(origin),
        direction(direction),
        depth(depth) {}
  __device__ GPURay(const float3 &origin, const float3 &direction,
                    const GPUStack<float> &refraction_coeffs, int depth)
       : origin(origin),
         direction(direction),
         refraction_coeffs(refraction_coeffs),
         depth(depth) {}
  __device__ ~GPURay() {}

  float3 origin;
  float3 direction;
  // TODO(Mateus): this is not quite right - two different objects may have the same refraction
  // coefficient. Fix this on a later version (we found some issues with the inheritance solution
  // used on CPU's implementation).
  GPUStack<float> refraction_coeffs;
  int depth;
};

}  //namespace gpu

#endif  // GPU_RAY_H_

#ifndef GPU_CAMERA_H_
#define GPU_CAMERA_H_

#include <cuda_runtime.h>

namespace gpu {

struct GPUCamera {
 public:
  __host__ __device__ GPUCamera() : width(0), height(0) {}
  __host__ __device__ GPUCamera(const float3 &eye, const float2 &bottom, const float2 &top,
                                int width, int height)
      : eye(eye),
        bottom(bottom),
        top(top),
        width(width),
        height(height) {}
  __host__ __device__ ~GPUCamera() {}

  float3 eye;
  float2 bottom;
  float2 top;

  int width;
  int height;
};

}  // namespace gpu

#endif  // GPU_CAMERA_H_

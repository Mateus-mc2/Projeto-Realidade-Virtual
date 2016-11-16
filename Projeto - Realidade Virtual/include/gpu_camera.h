#ifndef GPU_CAMERA_H_
#define GPU_CAMERA_H_

#include <cuda_runtime.h>

namespace gpu {

struct GPUCamera {
 public:
  __device__ GPUCamera() {}
  __device__ GPUCamera(const float3 &eye, const float2 &bottom, const float2 &top,
                       float width, float height)
      : eye(eye),
        bottom(bottom),
        top(top),
        width(width),
        height(height) {}
  __device__ ~GPUCamera() {}

  float3 eye;
  float2 bottom;
  float2 top;

  float width;
  float height;
};

}  // namespace gpu

#endif  // GPU_CAMERA_H_

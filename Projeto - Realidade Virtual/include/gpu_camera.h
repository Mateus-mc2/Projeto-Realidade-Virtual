#ifndef GPU_CAMERA_H_
#define GPU_CAMERA_H_

#include <cuda_runtime.h>

#include "camera.h"

namespace gpu {

struct GPUCamera {
 public:
  __host__ __device__ GPUCamera() {}
  __host__ __device__ GPUCamera(const util::Camera &cpu_camera)
      : eye(make_float3(cpu_camera.eye.x(), cpu_camera.eye.y(), cpu_camera.eye.z())),
        bottom(make_float2(cpu_camera.bottom.x(), cpu_camera.bottom.y())),
        top(make_float2(cpu_camera.top.x(), cpu_camera.top.y())),
        width(cpu_camera.width),
        height(cpu_camera.height) {}
  __host__ __device__ GPUCamera(const float3 &eye, const float2 &bottom, const float2 &top,
                                float width, float height)
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

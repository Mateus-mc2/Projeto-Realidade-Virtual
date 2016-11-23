#ifndef GPU_LIGHT_H_
#define GPU_LIGHT_H_

#include <iostream>

#include <cuda_runtime.h>

namespace gpu {

// Point particle light struct for GPU.
struct GPULight {
 public:
  GPULight() {}
  GPULight(const float3 &position, float red, float green, float blue, float intensity)
      : position(position),
        red(red),
        green(green),
        blue(blue),
        intensity(intensity) {}

  ~GPULight() {}

  float3 position;

  float red;
  float green;
  float blue;

  float intensity;
};

}  // namespace gpu

#endif  // GPU_LIGHT_H_

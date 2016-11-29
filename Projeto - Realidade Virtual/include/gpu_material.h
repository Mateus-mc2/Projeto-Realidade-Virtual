#ifndef GPU_MATERIAL_H_
#define GPU_MATERIAL_H_

#include <cuda_runtime.h>

namespace gpu {

struct GPUMaterial {
 public:
  __host__ __device__ GPUMaterial() {}
  __host__ __device__ GPUMaterial(float3 color, float refraction_coeff, float k_a, float k_d,
                                  float k_s, float k_t, float n)
      : red(color.x),
        green(color.y),
        blue(color.z),
        refraction_coeff(refraction_coeff),
        k_a(k_a),
        k_d(k_d),
        k_s(k_s),
        k_t(k_t),
        n(n) {}
  __host__ __device__ ~GPUMaterial() {}

  float red;   // Red component.
  float green; // Green component.
  float blue;  // Blue component.
  float refraction_coeff;  // Refraction coefficient.

  float k_a;  // Ambient reflection coefficient.
  float k_d;  // Diffusion reflection coefficient.
  float k_s;  // Specular reflection coefficient.
  float k_t;  // Transparency coefficient.
  float n;    // Specular reflection exponent.
};

}  // namespace gpu

#endif  // MATERIAL_H_

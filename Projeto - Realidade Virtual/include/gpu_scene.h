#ifndef GPU_SCENE_H_
#define GPU_SCENE_H_

#include <iostream>

#include <cuda_runtime.h>
#include <thrust/device_free.h>
#include <thrust/device_ptr.h>

#include "gpu_camera.h"
#include "gpu_light.h"
#include "gpu_quadric.h"
#include "gpu_triangular_object.h"
#include "gpu_vector.h"

namespace gpu {

struct GPUScene {
 public:
  GPUScene() {}
  GPUScene(const GPUCamera &camera, const float3 &bg_color, float ambient_light_intensity,
           const GPUVector<GPULight> &lights, const GPUVector<GPUQuadric> &quadrics,
           const GPUVector<GPUTriangularObject> &objs, bool use_anti_aliasing, int num_paths,
           int max_depth, float tone_mapping, int seed, int light_sampling_type)
      : camera(camera),
        bg_color(bg_color),
        lights(lights),
        quadrics(quadrics),
        triangular_objs(objs),
        ambient_light_intensity(ambient_light_intensity),
        tone_mapping(tone_mapping),
        use_anti_aliasing(use_anti_aliasing),
        num_paths(num_paths),
        max_depth(max_depth),
        seed(seed),
        light_sampling_type(light_sampling_type) {}

  ~GPUScene() {}

  GPUCamera camera;
  float3 bg_color;

  GPUVector<GPULight> lights;
  GPUVector<GPUQuadric> quadrics;
  GPUVector<GPUTriangularObject> triangular_objs;

  float ambient_light_intensity;
  float tone_mapping;

  bool use_anti_aliasing;

  int num_paths;
  int max_depth;
  int seed;
  int light_sampling_type;

  float max_float;
};

}  // namespace gpu

#endif  // GPU_SCENE_H_

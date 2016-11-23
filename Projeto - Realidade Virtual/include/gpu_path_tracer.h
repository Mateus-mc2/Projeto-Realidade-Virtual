#ifndef GPU_PATH_TRACER_H_
#define GPU_PATH_TRACER_H_

#include <cuda_runtime.h>
#include <opencv2/core/core.hpp>
#include <thrust/random.h>

#include "gpu_ray.h"
#include "gpu_scene.h"

namespace gpu {

struct GPUPathTracer {
 public:
  explicit GPUPathTracer(int seed)
      : generator(seed),
        distribution(0.0f, 1.0f) {}

  cv::Mat RenderScene(const GPUScene &scene);

  thrust::minstd_rand generator;
  thrust::uniform_real_distribution<float> distribution;
};

}  // namespace gpu

#endif  // GPU_PATH_TRACER_H_

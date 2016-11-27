#ifndef GPU_PATH_TRACER_H_
#define GPU_PATH_TRACER_H_

#include <cuda_runtime.h>
#include <opencv2/core/core.hpp>
#include <thrust/device_free.h>
#include <thrust/device_new.h>
#include <thrust/device_ptr.h>
#include <thrust/random.h>

#include "gpu_ray.h"
#include "gpu_scene.h"

namespace gpu {

struct GPUPathTracer {
 public:
  explicit GPUPathTracer(int seed) {
    cudaMallocManaged(&this->generator, sizeof(thrust::default_random_engine));
    cudaMallocManaged(&this->distribution, sizeof(thrust::uniform_real_distribution<float>));

    this->generator->seed(seed);
  }

  ~GPUPathTracer() {
    cudaFree(this->generator);
    cudaFree(this->distribution);
  }

  cv::Mat RenderScene(const GPUScene *scene);

  thrust::default_random_engine *generator;
  thrust::uniform_real_distribution<float> *distribution;
};

}  // namespace gpu

#endif  // GPU_PATH_TRACER_H_

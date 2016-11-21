#ifndef GPU_PATH_TRACER_H_
#define GPU_PATH_TRACER_H_

#include <cuda_runtime.h>
#include <opencv2/core/core.hpp>
#include <thrust/random.h>

#include "gpu_ray.h"
#include "gpu_scene.h"

class GPUPathTracer {
 public:
  explicit GPUPathTracer(const GPUScene &scene)
      : scene_(scene),
        generator_(scene.seed),
        distribution_(0.0f, 1.0f) {}

  cv::Mat RenderScene();

 private:
  __global__ void KernelLaunch(uchar *img_data);
  __device__ float3 TracePath(const GPURay &ray);

  GPUScene scene_;
  thrust::minstd_rand generator_;
  thrust::uniform_real_distribution<float> distribution_;
};

#endif  // GPU_PATH_TRACER_H_

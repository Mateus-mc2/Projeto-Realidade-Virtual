#ifndef GPU_PATH_TRACER_H_
#define GPU_PATH_TRACER_H_

#include <cuda_runtime.h>

#include "gpu_ray.h"

class GPUPathTracer {
 public:

 private:
  __global__ float3 TracePath(const GPURay &ray);
};

#endif  // GPU_PATH_TRACER_H_

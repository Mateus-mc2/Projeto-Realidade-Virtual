#include "gpu_path_tracer.h"

#include <iostream>

#include <device_launch_parameters.h>
#include <opencv2/ximgproc/edge_filter.hpp>

namespace gpu {
namespace {

const double kSigmaS = 12.401;
const double kSigmaR = 0.8102;
const bool kAdjustOutliers = true;

const int kNumThreads = 1 << 8;
typedef cv::Vec<float, 6> GeometricInfo;

__global__ void KernelLaunch(const GPUPathTracer &path_tracer, cv::Vec3f *img_data,
                             GeometricInfo *geometric_info_data, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    
  }
}

__device__ float3 TracePath(const GPUPathTracer &path_tracer, const GPURay &ray);

}  // namespace

cv::Mat GPUPathTracer::RenderScene(const GPUScene &scene) {
  int rows = scene.camera.height;
  int cols = scene.camera.width;

  cv::Vec3f *img_data;
  GeometricInfo *geometric_info_data;

  cudaError_t error;

  error = cudaMallocManaged(&img_data, rows * cols);
  if (error != cudaSuccess) {
    std::cout << "Cuda error " << error << std::endl;
    return cv::Mat();
  }

  error = cudaMallocManaged(&geometric_info_data, rows * cols);
  if (error != cudaSuccess) {
    std::cout << "Cuda error " << error << std::endl;
    return cv::Mat();
  }

  int num_blocks = rows * cols / kNumThreads + 1;
  KernelLaunch<<<num_blocks, kNumThreads>>>(*this, img_data, geometric_info_data,
                                            rows * cols);
  cudaDeviceSynchronize();

  cv::Mat rendered_img(rows, cols, CV_32FC3);
  cv::Mat geometric_info(rows, cols, CV_32FC(6));

  //memcpy(rendered_img.data, reinterpret_cast<uchar*>(img_data), sizeof(cv::Vec3f) * rows * cols);
  //memcpy(geometric_info.data, reinterpret_cast<uchar*>(geometric_info_data),
  //       sizeof(GeometricInfo) * rows * cols);

  cudaFree(img_data);
  cudaFree(geometric_info_data);

  //if (rows * cols > 0)
  //  cv::ximgproc::amFilter(geometric_info, rendered_img, rendered_img, kSigmaS, kSigmaR,
  //                         kAdjustOutliers);

  return rendered_img;
}

}  // namespace gpu

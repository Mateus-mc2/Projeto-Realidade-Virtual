#include "gpu_path_tracer.h"

#include <iostream>

#include <device_launch_parameters.h>
#include <opencv2/ximgproc/edge_filter.hpp>
#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_new.h>
#include <thrust/device_ptr.h>

#include "math_lib.h"

namespace gpu {
namespace {

const double kSigmaS = 12.401;
const double kSigmaR = 0.8102;
const bool kAdjustOutliers = true;

const int kNumThreads = 1 << 8;

struct GeometricInfo {
  float3 position;
  float3 normal;
};

//__device__ __managed__ float3 *img_data;
//__device__ __managed__ GeometricInfo *geometric_info_data;

__device__ void GetNearestObjectAndIntersection(GPUScene *scene, const GPURay &ray,
                                                gpu::GPURenderableObject **object,
                                                float *param, float3 *normal) {
  *param = -1.0f;
  float3 curr_normal;

  for (int i = 0; i < scene->quadrics.size(); ++i) {
    float t = scene->quadrics[i].GetIntersectionParameter(ray, &curr_normal);

    if (*param < 0.0f || (t > 0.0f && *param > t)) {
      *param = t;
      *object = &scene->quadrics[i];
      *normal = curr_normal;
    }
  }

  for (int i = 0; i < scene->triangular_objs.size(); ++i) {
    float t = scene->triangular_objs[i].GetIntersectionParameter(ray, &curr_normal);

    if (*param < 0.0f || (t > 0.0f && *param > t)) {
      *param = t;
      *object = &scene->quadrics[i];
      *normal = curr_normal;
    }
  }
}

__device__ float3 TracePath(GPUScene *scene, const thrust::minstd_rand &generator,
                            thrust::uniform_real_distribution<float> &distribution,
                            const GPURay &ray) {
  float3 color = make_float3(0.0f, 0.0f, 0.0f);

  return color;
}

__global__ void KernelLaunch(GPUScene *scene, thrust::minstd_rand generator,
                             thrust::uniform_real_distribution<float> distribution,
                             int size, float3 *img_data, GeometricInfo *geometric_info_data) {
  int i = blockIdx.x;
  int j = threadIdx.x;
  int idx = i * blockDim.x + j;

  if (idx < size) {
    float pixel_width = (scene->camera.top.x - scene->camera.bottom.x) / scene->camera.width;
    float pixel_height = (scene->camera.top.y - scene->camera.bottom.y) / scene->camera.height;

    float x_t = scene->use_anti_aliasing ? distribution(generator) : 0.5f;
    float y_t = scene->use_anti_aliasing ? distribution(generator) : 0.5f;
    float3 looking_at, direction;

    looking_at.x = scene->camera.bottom.x + x_t*pixel_width + j*pixel_width;
    looking_at.y = scene->camera.top.x - y_t*pixel_height - i*pixel_height;
    looking_at.z = 0.0;

    direction.x = looking_at.x - scene->camera.eye.x;
    direction.y = looking_at.y - scene->camera.eye.y;
    direction.z = looking_at.z - scene->camera.eye.z;

    math::Normalize(&direction);

    gpu::GPURay ray(scene->camera.eye, direction);
    float3 color = TracePath(scene, generator, distribution, ray);

    img_data[idx].x += color.x;
    img_data[idx].y += color.y;
    img_data[idx].z += color.z;

    // Gather geometric information.
    gpu::GPURenderableObject *object;
    float t;

    GetNearestObjectAndIntersection(scene, ray, &object, &t, &geometric_info_data[idx].normal);

    geometric_info_data[idx].position.x = ray.origin.x + t * ray.direction.x;
    geometric_info_data[idx].position.y = ray.origin.y + t * ray.direction.y;
    geometric_info_data[idx].position.z = ray.origin.z + t * ray.direction.z;
  }
}

}  // namespace

cv::Mat GPUPathTracer::RenderScene(const GPUScene &scene) {
  int rows = scene.camera.height;
  int cols = scene.camera.width;

  float3 *img_data;
  GeometricInfo *geometric_info_data;
  cudaError_t error;

  error = cudaMallocManaged(&img_data, sizeof(float3) * rows * cols);
  if (error != cudaSuccess) {
    std::cout << "Cuda error " << error << std::endl;
    return cv::Mat();
  }

  error = cudaMallocManaged(&geometric_info_data, sizeof(GeometricInfo) * rows * cols);
  if (error != cudaSuccess) {
    std::cout << "Cuda error " << error << std::endl;
    return cv::Mat();
  }

  std::cout << img_data[0].x << std::endl;
  std::cout << geometric_info_data[0].position.x << std::endl;

  int num_blocks = rows * cols / kNumThreads + 1;
  thrust::device_ptr<GPUScene> d_ptr = thrust::device_malloc<GPUScene>(sizeof(GPUScene));
  d_ptr = thrust::device_new<GPUScene>(d_ptr, scene);

  KernelLaunch<<<num_blocks, kNumThreads>>>(thrust::raw_pointer_cast(scene), this->generator,
                                            this->distribution, rows * cols, img_data,
                                            geometric_info_data);
  cudaDeviceSynchronize();

  thrust::device_free(d_ptr);

  cv::Mat rendered_img(rows, cols, CV_32FC3);
  cv::Mat geometric_info(rows, cols, CV_32FC(6));

  // Copy pointers contents to matrices.
#pragma omp parallel for
  for (int i = 0; i < rows; ++i) {
    cv::Vec3f *img_row_ptr = rendered_img.ptr<cv::Vec3f>(i);
    cv::Vec<float, 6> *info_row_ptr = geometric_info.ptr<cv::Vec<float, 6>>(i);

    for (int j = 0; j < cols; ++j) {
      int idx = i * cols + j;

      img_row_ptr[j][0] = img_data[idx].z;
      img_row_ptr[j][1] = img_data[idx].y;
      img_row_ptr[j][2] = img_data[idx].x;

      info_row_ptr[j][0] = geometric_info_data[idx].position.x;
      info_row_ptr[j][1] = geometric_info_data[idx].position.y;
      info_row_ptr[j][2] = geometric_info_data[idx].position.z;

      info_row_ptr[j][3] = geometric_info_data[idx].normal.x;
      info_row_ptr[j][4] = geometric_info_data[idx].normal.x;
      info_row_ptr[j][5] = geometric_info_data[idx].normal.x;
    }
  }

  cudaFree(img_data);
  cudaFree(geometric_info_data);

  rendered_img /= scene.num_paths;
  //if (rows * cols > 0)
  //  cv::ximgproc::amFilter(geometric_info, rendered_img, rendered_img, kSigmaS, kSigmaR,
  //                         kAdjustOutliers);

  return rendered_img;
}

}  // namespace gpu

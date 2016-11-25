#include "gpu_path_tracer.h"

#include <iostream>

#include <device_launch_parameters.h>
#include <opencv2/ximgproc/edge_filter.hpp>
#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_new.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

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


__device__ void GetNearestObjectAndIntersection(const GPURay &ray, const GPUScene *scene,
                                                float *param, float3 *normal,
                                                gpu::GPUTriangularObject *object) {
  *param = -1.0f;
  float3 curr_normal;

  for (int i = 0; i < scene->triangular_objs.size(); ++i) {
    float t = scene->triangular_objs[i].GetIntersectionParameter(ray, &curr_normal);

    if (*param < 0.0f || (*param > t && t > 0.0f)) {
      *object = scene->triangular_objs[i];
      *param = t;
      *normal = curr_normal;
    }
  }
}

__device__ void GetNearestObjectAndIntersection(const GPURay &ray, const GPUScene *scene,
                                                float *param, float3 *normal,
                                                gpu::GPUQuadric *object) {
  *param = -1.0f;
  float3 curr_normal;

  for (int i = 0; i < scene->quadrics.size(); ++i) {
    float t = scene->quadrics[i].GetIntersectionParameter(ray, &curr_normal);

    if (*param < 0.0f || (*param > t && t > 0.0f)) {
      *object = scene->quadrics[i];
      *param = t;
      *normal = curr_normal;
    }
  }
}

__device__ float3 TracePath(const GPUScene *scene, thrust::minstd_rand *generator,
                            thrust::uniform_real_distribution<float> *distribution,
                            const GPURay &ray) {
  float3 color = make_float3(1.0f, 1.0f, 1.0f);

  return color;
}

__global__ void LaunchKernel(const GPUScene *scene, thrust::minstd_rand *generator,
                             thrust::uniform_real_distribution<float> *distribution,
                             int size, float3 *img_data, GeometricInfo *geometric_info_data) {
  int i = blockIdx.x;
  int j = threadIdx.x;
  int idx = i * blockDim.x + j;

  if (idx < size) {
    float pixel_width = (scene->camera.top.x - scene->camera.bottom.x) / scene->camera.width;
    float pixel_height = (scene->camera.top.y - scene->camera.bottom.y) / scene->camera.height;

    float x_t = scene->use_anti_aliasing ? (*distribution)(*generator) : 0.5f;
    float y_t = scene->use_anti_aliasing ? (*distribution)(*generator) : 0.5f;
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
    gpu::GPUQuadric quadric;
    gpu::GPUTriangularObject obj;
    float t1, t2;
    float3 normal1, normal2;

    GetNearestObjectAndIntersection(ray, scene, &t1, &normal1, &quadric);
    GetNearestObjectAndIntersection(ray, scene, &t2, &normal2, &obj);

    float t = t1 < t2 ? t1 : t2;
    geometric_info_data[idx].normal = t1 < t2 ? normal1 : normal2;

    geometric_info_data[idx].position.x = ray.origin.x + t * ray.direction.x;
    geometric_info_data[idx].position.y = ray.origin.y + t * ray.direction.y;
    geometric_info_data[idx].position.z = ray.origin.z + t * ray.direction.z;
  }
}

}  // namespace

cv::Mat GPUPathTracer::RenderScene(const GPUScene &scene) {
  int rows = scene.camera.height;
  int cols = scene.camera.width;

  cudaError_t error;
  float3 *img_data;
  GeometricInfo *geometric_info_data;

  error = cudaMallocManaged(&img_data, sizeof(float3) * rows * cols);
  if (error != cudaSuccess) {
    std::cerr << "Cuda error " << error << std::endl;
    return cv::Mat();
  }

  error = cudaMallocManaged(&geometric_info_data, sizeof(GeometricInfo) * rows * cols);
  if (error != cudaSuccess) {
    std::cerr << "Cuda error " << error << std::endl;
    return cv::Mat();
  }

  GPUScene *device_scene;
  error = cudaMalloc(&device_scene, sizeof(GPUScene));
  if (error != cudaSuccess) {
    std::cerr << "Cuda error " << error << std::endl;
    return cv::Mat();
  }

  error = cudaMemcpy(device_scene, &scene, sizeof(GPUScene), cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    std::cerr << "Cuda error " << error << std::endl;
    return cv::Mat();
  }

  int num_blocks = rows * cols / kNumThreads + 1;
  LaunchKernel<<<num_blocks, kNumThreads>>>(device_scene, this->generator, this->distribution,
                                            rows * cols, img_data, geometric_info_data);
  cudaDeviceSynchronize();
  cudaFree(device_scene);

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

  rendered_img /= scene.num_paths;
  //if (rows * cols > 0)
  //  cv::ximgproc::amFilter(geometric_info, rendered_img, rendered_img, kSigmaS, kSigmaR,
  //                         kAdjustOutliers);

  return rendered_img;
}

}  // namespace gpu

#include "gpu_path_tracer.h"

#include <iostream>

#include <device_launch_parameters.h>
#include <opencv2/ximgproc/edge_filter.hpp>

#include "math_lib.h"

namespace gpu {
namespace {

const double kSigmaS = 12.401;
const double kSigmaR = 0.8102;
const bool kAdjustOutliers = true;

__device__ const float kEps = 1.0e-03f;

const int kNumThreads = 1 << 8;

struct GeometricInfo {
  float3 position;
  float3 normal;
};


__device__ GPUMatrix ComputeBaseFromNormal(const float3 &normal) {
  float3 u, v;
  if (abs(normal.x) > abs(normal.y)) {
    u.x = normal.z;
    u.y = 0.0f;
    u.z = -normal.x;
  } else {
    u.x = 0.0f;
    u.y = -normal.y;
    u.z = normal.z;
  }

  math::Normalize(&u);
  v = math::Cross(normal, u);
  GPUMatrix base_transform(3, 3);

  base_transform(0, 0) = normal.x;
  base_transform(0, 1) = normal.y;
  base_transform(0, 2) = normal.z;

  base_transform(1, 0) = u.x;
  base_transform(1, 1) = u.y;
  base_transform(1, 2) = u.z;

  base_transform(2, 0) = v.x;
  base_transform(2, 1) = v.y;
  base_transform(2, 2) = v.z;

  return base_transform;
}

__device__ void GetNearestObjectAndIntersection(const GPURay &ray, const GPUScene *scene,
                                                float *param, float3 *normal,
                                                gpu::GPUTriangularObject **object) {
  *param = scene->max_float;
  float3 curr_normal;

  for (int i = 0; i < scene->triangular_objs.size(); ++i) {
    float t = scene->triangular_objs[i].GetIntersectionParameter(ray, scene->max_float,
                                                                 &curr_normal);

    if (*param > t && t > 0.0f) {
      *object = &scene->triangular_objs[i];
      *param = t;
      *normal = curr_normal;
    }
  }
}

__device__ void GetNearestObjectAndIntersection(const GPURay &ray, const GPUScene *scene,
                                                float *param, float3 *normal,
                                                gpu::GPUQuadric **object) {
  *param = scene->max_float;
  float3 curr_normal;

  for (int i = 0; i < scene->quadrics.size(); ++i) {
    float t = scene->quadrics[i].GetIntersectionParameter(ray, &curr_normal);

    if (*param > t && t > 0.0f) {
      *object = &scene->quadrics[i];
      *param = t;
      *normal = curr_normal;
    }
  }
}

__device__ float Illuminate(const GPURay &shadow_ray, float3 light_position,
                            const GPUScene *scene, int light_index) {
  float final_intensity = scene->lights[light_index].intensity;
  float3 normal;

  const float max_t = shadow_ray.direction.x != 0 ?
                     (light_position.x - shadow_ray.origin.x) / shadow_ray.direction.x :
                     (shadow_ray.direction.y != 0 ?
                     (light_position.y - shadow_ray.origin.y) / shadow_ray.direction.y :
                     (light_position.z - shadow_ray.origin.z) / shadow_ray.direction.z);

  for (int i = 0; i < scene->quadrics.size() && final_intensity > 0.0f; ++i) {
    const GPUMaterial &obj_material = scene->quadrics[i].material();
    float t = scene->quadrics[i].GetIntersectionParameter(shadow_ray, &normal);

    if (t > 0.0 && t < max_t) {
      final_intensity *= obj_material.k_t;
    }
  }

  for (int i = 0; i < scene->triangular_objs.size() && final_intensity > 0; ++i) {
    const GPUMaterial &obj_material = scene->quadrics[i].material();
    float t = scene->triangular_objs[i].GetIntersectionParameter(shadow_ray, scene->max_float,
                                                                 &normal);

    if (t > 0.0 && t < max_t) {
      final_intensity *= obj_material.k_t;
    }
  }

  return final_intensity;
}

__device__ float3 TracePath(const GPUScene *scene, thrust::default_random_engine *generator,
                            thrust::uniform_real_distribution<float> *distribution,
                            GPURay &ray) {
  float3 color = make_float3(0.0f, 0.0f, 0.0f);
  float3 position, viewer;
  const float kPi = acos(-1.0f);

  float accumulated_scalar_factor = 1.0f;

  while (ray.depth <= scene->max_depth) {
    GPUQuadric *quadric = nullptr;
    GPUTriangularObject *obj = nullptr;
    float t1 = -1.0f, t2 = -1.0f;
    float3 normal1, normal2;

    GetNearestObjectAndIntersection(ray, scene, &t1, &normal1, &quadric);
    GetNearestObjectAndIntersection(ray, scene, &t2, &normal2, &obj);

    if (!quadric && !obj) {
      // No intersection.
      color.x += accumulated_scalar_factor * scene->bg_color.x;
      color.y += accumulated_scalar_factor * scene->bg_color.y;
      color.z += accumulated_scalar_factor * scene->bg_color.z;

      break;
    }

    bool has_quadric_intersection = !math::IsAlmostEqual(t1, -1.0f, kEps);
    float t = (t1 < t2 && has_quadric_intersection) ? t1 : t2;
    float3 normal = (t1 < t2 && has_quadric_intersection) ? normal1 : normal2;

    position.x = ray.origin.x + t * ray.direction.x;
    position.y = ray.origin.y + t * ray.direction.y;
    position.z = ray.origin.z + t * ray.direction.z;

    viewer.x = ray.origin.x - position.x;
    viewer.y = ray.origin.y - position.y;
    viewer.z = ray.origin.z - position.z;

    math::Normalize(&viewer);

    const GPUMaterial &material = (t1 < t2 && has_quadric_intersection) ? quadric->material() :
                                                                          obj->material();
    float3 material_color = make_float3(material.red, material.green, material.blue);
    float3 curr_color = material_color;

    curr_color.x *= material.k_a * scene->ambient_light_intensity;
    curr_color.y *= material.k_a * scene->ambient_light_intensity;
    curr_color.z *= material.k_a * scene->ambient_light_intensity;

    float3 direct_contrib = make_float3(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < scene->lights.size(); ++i) {
      float3 light_position = scene->lights[i].position;
      float3 light_direction;

      light_direction.x = light_position.x - position.x;
      light_direction.y = light_position.y - position.y;
      light_direction.z = light_position.z - position.z;

      math::Normalize(&light_direction);
      float cos_theta = math::InnerProduct(normal, light_direction);

      if (cos_theta > 0.0f) {
        GPURay shadow_ray(position, light_direction);
        float light_intensity = Illuminate(shadow_ray, light_position, scene, i);

        float3 reflected;
        reflected.x = 2 * normal.x * cos_theta - light_direction.x;
        reflected.y = 2 * normal.y * cos_theta - light_direction.y;
        reflected.z = 2 * normal.z * cos_theta - light_direction.z;

        math::Normalize(&reflected);

        float cos_alpha = math::InnerProduct(reflected, viewer);
        float pow_by_spec_exp = powf(cos_alpha, material.n);
        float3 light_color;

        light_color.x = scene->lights[i].red;
        light_color.y = scene->lights[i].green;
        light_color.z = scene->lights[i].blue;

        float3 dir_prod = math::DirectProduct3(material_color, light_color);

        direct_contrib.x += light_intensity * (material.k_d * dir_prod.x * cos_theta +
                                               material.k_s * light_color.x * pow_by_spec_exp);
        direct_contrib.y += light_intensity * (material.k_d * dir_prod.y * cos_theta +
                                               material.k_s * light_color.y * pow_by_spec_exp);
        direct_contrib.z += light_intensity * (material.k_d * dir_prod.z * cos_theta +
                                               material.k_s * light_color.z * pow_by_spec_exp);
      }
    }

    curr_color.x += direct_contrib.x;
    curr_color.y += direct_contrib.y;
    curr_color.z += direct_contrib.z;

    color.x += accumulated_scalar_factor * curr_color.x;
    color.y += accumulated_scalar_factor * curr_color.y;
    color.z += accumulated_scalar_factor * curr_color.z;

    // Indirect component - get on next iteration.
    float k_total = material.k_d + material.k_s + material.k_t;
    float ray_type = (*distribution)(*generator) * k_total;

    if (ray_type < material.k_d) {
      // Generate a ray with random direction with origin on intesected point (using uniform sphere distribution here).
      float r_1 = (*distribution)(*generator);
      float r_2 = (*distribution)(*generator);
      float phi = acosf(std::sqrt(r_1));
      float theta = 2 * kPi * r_2;

      GPUMatrix uniform_sample(3, 1);

      uniform_sample(0, 0) = sinf(phi) * cosf(theta);
      uniform_sample(1, 0) = sinf(phi) * sinf(theta);
      uniform_sample(2, 0) = cosf(phi);

      GPUMatrix T = ComputeBaseFromNormal(normal);
      uniform_sample = T * uniform_sample;
      float3 rand_direction = make_float3(uniform_sample(0, 0), uniform_sample(1, 0),
                                          uniform_sample(2, 0));

      math::Normalize(&rand_direction);
      ray.origin = position;
      ray.direction = rand_direction;
      ++ray.depth;

      accumulated_scalar_factor *= material.k_d;
    } else if (ray_type < material.k_d + material.k_s) {
      float cos_theta = math::InnerProduct(normal, viewer);
      float3 reflected;

      reflected.x = 2 * normal.x * cos_theta - viewer.x;
      reflected.y = 2 * normal.y * cos_theta - viewer.y;
      reflected.z = 2 * normal.z * cos_theta - viewer.z;

      math::Normalize(&reflected);

      ray.origin = position;
      ray.direction = reflected;
      ++ray.depth;

      accumulated_scalar_factor *= material.k_s;
    } else {
      float n_1, n_2;

      if (ray.refraction_coeffs.IsEmpty()) {
        // Scene's ambient refraction coefficient (we're assuming n = 1.0 here).
        n_1 = 1.0f;
        n_2 = material.refraction_coeff;
        ray.refraction_coeffs.Push(n_2);
      } else {  // Ray is getting out of the object.
        n_1 = ray.refraction_coeffs.Peek();

        if (math::IsAlmostEqual(n_1, material.refraction_coeff, kEps)) {
          ray.refraction_coeffs.Pop();
          n_2 = ray.refraction_coeffs.IsEmpty() ? 1.0f : ray.refraction_coeffs.Peek();
        } else {
          n_2 = material.refraction_coeff;
          ray.refraction_coeffs.Push(n_2);
        }

        float3 ray_dir_add_inverse = make_float3(-ray.direction.x, -ray.direction.y,
                                                 -ray.direction.z);
        float cos_theta_incident = math::InnerProduct(normal, ray_dir_add_inverse);
        if (cos_theta_incident < 0.0f) {  // If normal was accidentally inverted.
          normal.x = -normal.x;
          normal.y = -normal.y;
          normal.z = -normal.z;

          cos_theta_incident = -cos_theta_incident;
        }

        float sin_theta_incident = sqrt(1 - cos_theta_incident * cos_theta_incident);
        float n_r = n_2 / n_1;

        if (sin_theta_incident < n_r) {
          // If total internal reflection does not occur, transmit ray.
          n_r = 1 / n_r;
          float squared_sin_theta_incident = sin_theta_incident * sin_theta_incident;
          float cos_theta_transmitted = sqrt(1 - n_r * n_r * squared_sin_theta_incident);
          float3 transmitted;

          transmitted.x = (n_r * cos_theta_incident - cos_theta_transmitted) * normal.x +
                           n_r * ray.direction.x;
          transmitted.y = (n_r * cos_theta_incident - cos_theta_transmitted) * normal.y +
                           n_r * ray.direction.y;
          transmitted.z = (n_r * cos_theta_incident - cos_theta_transmitted) * normal.z +
                           n_r * ray.direction.z;

          math::Normalize(&transmitted);
          ray.origin = position;
          ray.direction = transmitted;
          ++ray.depth;
        } else {
          float3 reflected;

          reflected.x = 2 * normal.x * cos_theta_incident + ray.direction.x;
          reflected.y = 2 * normal.y * cos_theta_incident + ray.direction.y;
          reflected.z = 2 * normal.z * cos_theta_incident + ray.direction.z;

          math::Normalize(&reflected);
          ray.origin = position;
          ray.direction = reflected;
          ++ray.depth;
        }

        accumulated_scalar_factor *= material.k_t;
      }
    }

    //color.x -= accumulated_scalar_factor * curr_color.x;
    //color.y -= accumulated_scalar_factor * curr_color.y;
    //color.z -= accumulated_scalar_factor * curr_color.z;
  }

  math::Clamp3(0.0f, 1.0f, &color);
  return color;
}

__global__ void LaunchKernel(const GPUScene *scene, int rows, int cols, float3 *img_data,
                             GeometricInfo *geometric_info_data) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < rows * cols) {
    thrust::default_random_engine generator(scene->seed);
    thrust::uniform_real_distribution<float> distribution;

    int i = static_cast<int>(floorf(static_cast<float>(idx) / cols));
    int j = idx % cols;

    //for (int k = 0; k < scene->num_paths; ++k) {
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

      GPURay ray(scene->camera.eye, direction);
      float3 color = TracePath(scene, &generator, &distribution, ray);

      img_data[idx].x += color.x;
      img_data[idx].y += color.y;
      img_data[idx].z += color.z;

      // Gather geometric information.
      GPUQuadric *quadric = nullptr;
      GPUTriangularObject *obj = nullptr;
      float t1, t2;
      float3 normal1, normal2;

      GetNearestObjectAndIntersection(ray, scene, &t1, &normal1, &quadric);
      GetNearestObjectAndIntersection(ray, scene, &t2, &normal2, &obj);

      if (!quadric && !obj && !(math::IsAlmostEqual(t1, -1.0f, kEps) ||
          math::IsAlmostEqual(t2, -1.0f, kEps))) {
        float t = t1 < t2 ? t1 : t2;
        geometric_info_data[idx].normal = t1 < t2 ? normal1 : normal2;

        geometric_info_data[idx].position.x += ray.origin.x + t * ray.direction.x;
        geometric_info_data[idx].position.y += ray.origin.y + t * ray.direction.y;
        geometric_info_data[idx].position.z += ray.origin.z + t * ray.direction.z;
      }
    //}

    //geometric_info_data[idx].position.x /= scene->num_paths;
    //geometric_info_data[idx].position.y /= scene->num_paths;
    //geometric_info_data[idx].position.z /= scene->num_paths;

    //geometric_info_data[idx].normal.x /= scene->num_paths;
    //geometric_info_data[idx].normal.y /= scene->num_paths;
    //geometric_info_data[idx].normal.z /= scene->num_paths;
  }
}

}  // namespace

cv::Mat GPUPathTracer::RenderScene(const GPUScene *scene) {
  int rows = scene->camera.height;
  int cols = scene->camera.width;

  cudaError_t error;
  float3 *img_data;
  GeometricInfo *geometric_info_data;

  error = cudaMallocManaged(&img_data, sizeof(float3) * rows * cols);
  if (error != cudaSuccess) {
    std::cerr << "Cuda error " << cudaGetErrorString(error) << std::endl;
    return cv::Mat();
  }

  error = cudaMallocManaged(&geometric_info_data, sizeof(GeometricInfo) * rows * cols);
  if (error != cudaSuccess) {
    std::cerr << "Cuda error " << cudaGetErrorString(error) << std::endl;
    return cv::Mat();
  }

  int num_blocks = rows * cols / kNumThreads + 1;
  LaunchKernel<<<rows, cols>>>(scene, rows, cols, img_data, geometric_info_data);
  error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    std::cerr << "Cuda error " << cudaGetErrorString(error) << std::endl;
    return cv::Mat();
  }

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

  rendered_img /= scene->num_paths;
  if (rows * cols > 0) {
    cv::ximgproc::amFilter(geometric_info, rendered_img, rendered_img, kSigmaS, kSigmaR,
                           kAdjustOutliers);
  }

  return rendered_img;
}

}  // namespace gpu

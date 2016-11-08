#include "pt_renderer.h"

#include <fstream>
#include <sstream>

#include <opencv2/ximgproc/edge_filter.hpp>

using Eigen::Vector2d;
using Eigen::Vector3d;

namespace pt {
namespace {

typedef cv::Vec<float, 6> GeometricInfo;

}

const double PTRenderer::kEps = 1.0e-03;

Vector3d PTRenderer::TracePath(const util::Ray &ray) {
  if (ray.depth > this->scene_.max_depth_) {
    return Vector3d(0.0, 0.0, 0.0);
  } else {
    util::RenderableObject *object = nullptr;
    double t;
    Vector3d normal;

    this->GetNearestObjectAndIntersection(ray, &object, &t, &normal);
    Vector3d intersection_point = ray.origin + t*ray.direction;

    if (math::IsAlmostEqual(t, -1.0, this->kEps) || object == nullptr) {
      return this->scene_.background_color_;
    }

    util::Material obj_material = object->material();
    Vector3d material_color(obj_material.red, obj_material.green, obj_material.blue);
    Vector3d color = this->scene_.ambient_light_intensity_*obj_material.k_a*material_color;

    Vector3d viewer = ray.origin - intersection_point;
    viewer = viewer / viewer.norm();

    // Check if the object hit is emissive
    if (object->emissive()) {
      return material_color;
    }

    //## COMPONENTE DIRETA
    //Vector3d partial_contribution(0,0,0);
    // Calcula-se a intensidade do objeto naquele ponto influenciada pelas fontes de luz na cena.
    //# Luzes pontuais
    for (int i = 0; i < this->scene_.point_lights_.size(); ++i) {
      Vector3d light_direction =  this->scene_.point_lights_[i].position - intersection_point;
      light_direction = light_direction / light_direction.norm();
      util::Ray shadow_ray(intersection_point, light_direction, ray.ambient_objs, 1);
      
      // Here we compute how much light is blocked by opaque and transparent surfaces, and use the
      // resulting intensity to scale diffuse and specular terms of the final color.
      double light_intensity = this->ScaleLightIntensity(this->scene_.point_lights_[i].intensity,
                                                         this->scene_.point_lights_[i].position,
                                                         shadow_ray);
      double cos_theta = normal.dot(light_direction);

      if (cos_theta > 0.0) {
        Vector3d reflected = 2*normal*cos_theta - light_direction;
        reflected = reflected / reflected.norm();
        
        double cos_alpha = reflected.dot(viewer);
        Vector3d light_color(this->scene_.point_lights_[i].red, 
                             this->scene_.point_lights_[i].green,
                             this->scene_.point_lights_[i].blue);

        color += light_intensity*(obj_material.k_d*material_color*cos_theta +
                                obj_material.k_s*light_color*std::pow(cos_theta, obj_material.n));
      }
    }

    //# Luzes extensas
    Vector3d light_direction;
    for (int i = 0; i < this->scene_.extense_lights_.size(); ++i) {
      for (int v = 0; v < this->scene_.extense_lights_[i].kVertices.size(); ++v) {
        light_direction =  this->scene_.extense_lights_[i].kVertices[v] - intersection_point;
        light_direction = light_direction / light_direction.norm();
        util::Ray shadow_ray(intersection_point, light_direction, ray.ambient_objs, 1);

        // Here we compute how much light is blocked by opaque and transparent surfaces, and use the
        // resulting intensity to scale diffuse and specular terms of the final color.
        double light_intensity = this->ScaleLightIntensity(this->scene_.extense_lights_[i].material().lp,
                                                           this->scene_.extense_lights_[i].kVertices[v],
                                                           shadow_ray);
        double cos_theta = normal.dot(light_direction);

        if (cos_theta > 0.0) {
          Vector3d reflected = 2*normal*cos_theta - light_direction;
          reflected = reflected / reflected.norm();
        
          double cos_alpha = reflected.dot(viewer);
          Vector3d light_color(this->scene_.extense_lights_[i].material().red,
                               this->scene_.extense_lights_[i].material().green,
                               this->scene_.extense_lights_[i].material().blue);

          color += light_intensity*(obj_material.k_d*material_color*cos_theta +
                                  obj_material.k_s*light_color*std::pow(cos_theta, obj_material.n));
        }
      }
    }


    // Para as areas dos triangulos
    Vector3d light_origin;
    Vector3d p1;
    Vector3d p2;
    Vector3d p3;
    Vector3d v1;
    Vector3d v2;

    if (this->scene_.lightsamplingtype_ == 1) {
      for (int i = 0; i < this->scene_.extense_lights_.size(); ++i) {
        double light_sampling_step = this->scene_.extense_lights_[i].material().light_sampling_step;
        // Itera para cada face, criando pontos de luz coerentes com a densidade da luz
        for (int f = 0; f < this->scene_.extense_lights_[i].kFaces.size(); ++f) {
          p1 = this->scene_.extense_lights_[i].kVertices[this->scene_.extense_lights_[i].kFaces[f][0]];
          p2 = this->scene_.extense_lights_[i].kVertices[this->scene_.extense_lights_[i].kFaces[f][1]];
          p3 = this->scene_.extense_lights_[i].kVertices[this->scene_.extense_lights_[i].kFaces[f][2]];
          v1 = p2 - p1;
          v2 = p3 - p1;
          double a_max = v1.norm();
          double b_max = v2.norm();
          double a_step = light_sampling_step/a_max;
          double b_step = light_sampling_step/b_max;

          for (double a = a_step; a < 1; a += a_step) {
            for (double b = b_step; b < 1; b += b_step) {
              light_origin = p1 + a*v1 + b*v2;
              light_direction =  light_origin - intersection_point;
              light_direction = light_direction / light_direction.norm();
              util::Ray shadow_ray(intersection_point, light_direction, ray.ambient_objs, 1);
      
              // Here we compute how much light is blocked by opaque and transparent surfaces, and use the
              // resulting intensity to scale diffuse and specular terms of the final color.
              double light_intensity = this->ScaleLightIntensity(this->scene_.extense_lights_[i].material().lp,
                                                                 light_origin,
                                                                 shadow_ray);
              double cos_theta = normal.dot(light_direction);

              if (cos_theta > 0.0) {
                Vector3d reflected = 2*normal*cos_theta - light_direction;
                reflected = reflected / reflected.norm();

                double cos_alpha = reflected.dot(viewer);
                Vector3d light_color(this->scene_.extense_lights_[i].material().red,
                                     this->scene_.extense_lights_[i].material().green,
                                     this->scene_.extense_lights_[i].material().blue);

                color += light_intensity*(obj_material.k_d*material_color*cos_theta +
                         obj_material.k_s*light_color*std::pow(cos_theta, obj_material.n));
              }
            }
          }
        }
      }
    } else if (this->scene_.lightsamplingtype_ == 2) {
      double area;
      int n_rays;
      for (int i = 0; i < this->scene_.extense_lights_.size(); ++i) {
        double light_density = this->scene_.extense_lights_[i].material().light_density;
        // Itera para cada face, criando pontos de luz coerentes com a densidade da luz
        for (int f = 0; f < this->scene_.extense_lights_[i].kFaces.size(); ++f) {
          p1 = this->scene_.extense_lights_[i].kVertices[this->scene_.extense_lights_[i].kFaces[f][0]];
          p2 = this->scene_.extense_lights_[i].kVertices[this->scene_.extense_lights_[i].kFaces[f][1]];
          p3 = this->scene_.extense_lights_[i].kVertices[this->scene_.extense_lights_[i].kFaces[f][2]];
          v1 = p2 - p1;
          v2 = p3 - p1;
          area = v1.cross(v2).norm()/2;
          n_rays = static_cast<int>(std::floor(area*light_density));

          for (int r = 0; r < n_rays; ++r) {
            double t1 = this->distribution_(this->light_generator_);
            double t2 = this->distribution_(this->light_generator_);
            light_origin = p1 + t1*v1 + t2*v2;
            light_direction =  light_origin - intersection_point;
            light_direction = light_direction / light_direction.norm();
            util::Ray shadow_ray(intersection_point, light_direction, ray.ambient_objs, 1);
      
            // Here we compute how much light is blocked by opaque and transparent surfaces, and use the
            // resulting intensity to scale diffuse and specular terms of the final color.
            double light_intensity = this->ScaleLightIntensity(this->scene_.extense_lights_[i].material().lp,
                                                               light_origin,
                                                               shadow_ray);
            double cos_theta = normal.dot(light_direction);

            if (cos_theta > 0.0) {
              Vector3d reflected = 2*normal*cos_theta - light_direction;
              reflected = reflected / reflected.norm();
        
              double cos_alpha = reflected.dot(viewer);
              Vector3d light_color(this->scene_.extense_lights_[i].material().red,
                                    this->scene_.extense_lights_[i].material().green,
                                    this->scene_.extense_lights_[i].material().blue);

              color += light_intensity*(obj_material.k_d*material_color*cos_theta +
                        obj_material.k_s*light_color*std::pow(cos_theta, obj_material.n));
            }
          }
        }
      }
    }

    //## COMPONENTE INDIRETA
    double k_tot = obj_material.k_d + obj_material.k_s + obj_material.k_t;
    double ray_type = this->distribution_(this->generator_)*k_tot;
    Vector3d indirect_light;

    if (ray_type < obj_material.k_d) {  // Throw a diffuse ray
      // Generate a ray with random direction with origin on intesected point (using uniform sphere distribution here).
      double r_1 = this->distribution_(this->ray_generator_);
      double r_2 = this->distribution_(this->ray_generator_);
      double phi = std::acos(std::sqrt(r_1));
      double theta = 2*M_PI*r_2;
      
      Vector3d rand_direction(std::sin(phi)*std::cos(theta),
                              std::sin(phi)*std::sin(theta),
                              std::cos(phi));
      util::Ray new_diffuse_ray(intersection_point, rand_direction, ray.ambient_objs, ray.depth + 1);
      indirect_light = obj_material.k_d * this->TracePath(new_diffuse_ray);
    } else if (ray_type < obj_material.k_d + obj_material.k_s) {  // Throw a specular ray
      Vector3d reflected = 2*normal*normal.dot(viewer) - viewer;
      reflected = reflected / reflected.norm();
      util::Ray new_specular_ray(intersection_point, reflected, ray.ambient_objs, ray.depth + 1);
      indirect_light = obj_material.k_s * this->TracePath(new_specular_ray);
    } else {  // Throw a refracted ray
      double n_1;
      double n_2;
      std::vector<util::RenderableObject*> objs_stack(ray.ambient_objs);

      if (objs_stack.empty()) {  // Scene's ambient refraction coefficient (we're assuming n = 1.0 here).
        n_1 = 1.0;
        n_2 = obj_material.refraction_coeff;
        objs_stack.push_back(object);
      } else {  // Ray is getting out of current object.
        util::RenderableObject *last_obj = objs_stack.back();
        n_1 = last_obj->material().refraction_coeff;

        if (object != last_obj) {
          n_2 = obj_material.refraction_coeff;
          objs_stack.push_back(object);
        } else {
          objs_stack.pop_back();
          n_2 = objs_stack.empty() ? 1.0 : objs_stack.back()->material().refraction_coeff;
        }
      }

      double cos_theta_incident = normal.dot(-ray.direction);
      
      if (cos_theta_incident < 0.0) {
        normal = -normal;
        cos_theta_incident = -cos_theta_incident;
      }

      double sin_theta_incident = std::sqrt(1 - cos_theta_incident*cos_theta_incident);

      // Check if it's a total internal reflection.
      if (sin_theta_incident < (n_2 / n_1)) {
        // Get new refracted ray.
        double n_r = n_1 / n_2;
        Vector3d refracted = (n_r*cos_theta_incident - std::sqrt(1 -
                              std::pow(n_r, 2)*std::pow(sin_theta_incident, 2)))*normal
                              + n_r*ray.direction;
        // Need to update the stack of objects.
        util::Ray new_refracted_ray(intersection_point, refracted, objs_stack, ray.depth + 1);
        indirect_light = obj_material.k_t * this->TracePath(new_refracted_ray);
      } else {  // Simulate total internal reflection.
        Vector3d total_internal_reflected = 2*normal*cos_theta_incident + ray.direction;
        util::Ray total_internal_reflection_ray(intersection_point,
                                                total_internal_reflected,
                                                ray.ambient_objs,
                                                ray.depth + 1);
        indirect_light = this->TracePath(total_internal_reflection_ray);
      }
    }

    color += indirect_light;

    // Clamp to guarantee that every ray returns a color r,g,b <= 1
    color[0] = std::min(color[0], 1.0);
    color[1] = std::min(color[1], 1.0);
    color[2] = std::min(color[2], 1.0);

    return color;
  }
}

//cv::Mat PTRenderer::GetImageGeometricInformation() {
//  int cols = static_cast<int>(this->scene_.camera_.width_);
//  int rows = static_cast<int>(this->scene_.camera_.height_);
//
//  cv::Mat result(rows, cols, CV_32FC(6));
//
//  double pixel_w = (this->scene_.camera_.top_.x() - this->scene_.camera_.bottom_.x()) /
//                    this->scene_.camera_.width_;
//  double pixel_h = (this->scene_.camera_.top_.y() - this->scene_.camera_.bottom_.y()) /
//                    this->scene_.camera_.height_;
//
//  for (int i = 0; i < rows; ++i) {
//    GeometricInfo *row_ptr = result.ptr<GeometricInfo>(i);
//
//    for (int j = 0; j < cols; ++j) {
//      double x_t = this->distribution_(this->anti_aliasing_generator_);
//      double y_t = this->distribution_(this->anti_aliasing_generator_);
//      Vector3d looking_at((this->scene_.camera_.bottom_.x() + x_t * pixel_w) + j * pixel_w,
//                          (this->scene_.camera_.top_.y() - y_t * pixel_h) - i * pixel_h, 0.0);
//      Vector3d direction = looking_at - this->scene_.camera_.eye_;
//      direction = direction / direction.norm();
//      util::Ray ray(this->scene_.camera_.eye_, direction, 1);
//
//      util::RenderableObject *object;
//      double t;
//      Vector3d normal;
//
//      this->GetNearestObjectAndIntersection(ray, &object, &t, &normal);
//      Vector3d intersection_point = ray.origin + t * ray.direction;
//
//      row_ptr[j][0] = static_cast<float>(intersection_point.x());
//      row_ptr[j][1] = static_cast<float>(intersection_point.y());
//      row_ptr[j][2] = static_cast<float>(intersection_point.z());
//
//      row_ptr[j][3] = static_cast<float>(normal.x());
//      row_ptr[j][4] = static_cast<float>(normal.y());
//      row_ptr[j][5] = static_cast<float>(normal.z());
//    }
//  }
//
//  return result;
//}

void PTRenderer::GetNearestObjectAndIntersection(const util::Ray &ray,
                                                 util::RenderableObject **object,
                                                 double *parameter,
                                                 Eigen::Vector3d *normal) {
  *parameter = std::numeric_limits<double>::max();
  Vector3d curr_normal;

  // Objetos descritos por quadrica
  for (int i = 0; i < this->scene_.quadrics_objects_.size(); ++i) {
    double curr_t = this->scene_.quadrics_objects_[i].GetIntersectionParameter(ray, &curr_normal);
    
    if (*parameter > curr_t && curr_t > this->kEps) {
      *object = &this->scene_.quadrics_objects_[i];
      *parameter = curr_t;
      *normal = curr_normal;
    }
  }

  // Objetos descritos por triangulos
  for (int i = 0; i < this->scene_.triangular_objects_.size(); ++i) {
    double curr_t = this->scene_.triangular_objects_[i].GetIntersectionParameter(ray, &curr_normal);

    if (*parameter > curr_t && curr_t > this->kEps) {
      *object = &this->scene_.triangular_objects_[i];
      *parameter = curr_t;
      *normal = curr_normal;
    }
  }

  // Objetos emissivos
  for (int i = 0; i < this->scene_.extense_lights_.size(); ++i) {
    double curr_t = this->scene_.extense_lights_[i].GetIntersectionParameter(ray, &curr_normal);

    if (*parameter > curr_t && curr_t > 0.0) {
      *object = &this->scene_.extense_lights_[i];
      *parameter = curr_t;
      *normal = curr_normal;
    }
  }
}


double PTRenderer::ScaleLightIntensity(double light_intensity,
                                       const Eigen::Vector3d &light_position,
                                       const util::Ray &shadow_ray) {
  double final_intensity = light_intensity;
  Vector3d normal;
  // Pega o índice do vetor diretor tal que v(idx) não seja zero. Ficou feioso assim, mas dane-se...
  const int idx = shadow_ray.direction(0) != 0 ? 0 : (shadow_ray.direction(1) != 0 ? 1 : 2);
  assert(shadow_ray.direction(idx) != 0);
  const double max_t = (light_position(idx) - shadow_ray.origin(idx)) / shadow_ray.direction(idx);

  for (int i = 0; i < this->scene_.quadrics_objects_.size() && final_intensity > 0; ++i) {
    const util::Material &obj_material = this->scene_.quadrics_objects_[i].material();
    double t = this->scene_.quadrics_objects_[i].GetIntersectionParameter(shadow_ray, &normal);

    if (t > 0.0 && t < max_t) {
      final_intensity *= obj_material.k_t;
    }
  }

  for (int i = 0; i < this->scene_.triangular_objects_.size() && final_intensity > 0; ++i) {
    const util::Material &obj_material = this->scene_.triangular_objects_[i].material();
    double t = this->scene_.triangular_objects_[i].GetIntersectionParameter(shadow_ray, &normal);

    if (t > 0.0 && t < max_t) {
      final_intensity *= obj_material.k_t;
    }
  }

  return final_intensity;
}

cv::Mat PTRenderer::RenderScene() {
  int rows = static_cast<int>(this->scene_.camera_.height_);
  int cols = static_cast<int>(this->scene_.camera_.width_);

  cv::Mat rendered_image(rows, cols, CV_64FC3);
  cv::Mat partial_result(rows, cols, CV_64FC3);
  cv::Mat geometric_information(rows, cols, CV_32FC(6));

  double pixel_w = (this->scene_.camera_.top_(0) - this->scene_.camera_.bottom_(0)) /
      this->scene_.camera_.width_;
  double pixel_h = (this->scene_.camera_.top_(1) - this->scene_.camera_.bottom_(1)) / 
      this->scene_.camera_.height_;
  int percent;
  int processed_rays = this->scene_.nmbr_paths_;

  if (this->scene_.antialiasing_) {  //\ Com anti-aliasing
    for (int i = 0; i < this->scene_.nmbr_paths_; ++i) {
      // Dispara um raio n vezes em um local randomico dentro do pixel
      #pragma omp parallel for
      for (int j = 0; j < rendered_image.rows; ++j) {
        cv::Vec3d *img_row_ptr = rendered_image.ptr<cv::Vec3d>(j);
        GeometricInfo *info_row_ptr = geometric_information.ptr<GeometricInfo>(j);

        for (int k = 0; k < rendered_image.cols; ++k) {
          double x_t = this->distribution_(this->anti_aliasing_generator_);
          double y_t = this->distribution_(this->anti_aliasing_generator_);
          Vector3d looking_at((this->scene_.camera_.bottom_(0) + x_t*pixel_w) + k*pixel_w,
                              (this->scene_.camera_.top_(1)    - y_t*pixel_h) - j*pixel_h,
                              0.0);
          Vector3d direction = looking_at - this->scene_.camera_.eye_;
          direction = direction / direction.norm();

          // Perform path tracing.
          util::Ray ray(this->scene_.camera_.eye_, direction, 1);
          Vector3d additional_color = this->TracePath(ray);

          img_row_ptr[k][0] += additional_color(2);
          img_row_ptr[k][1] += additional_color(1);
          img_row_ptr[k][2] += additional_color(0);

          // Gather geometric information.
          util::RenderableObject *object;
          double t;
          Vector3d normal;

          this->GetNearestObjectAndIntersection(ray, &object, &t, &normal);
          Vector3d intersection_point = ray.origin + t * ray.direction;

          info_row_ptr[k][0] = static_cast<float>(intersection_point.x());
          info_row_ptr[k][1] = static_cast<float>(intersection_point.y());
          info_row_ptr[k][2] = static_cast<float>(intersection_point.z());

          info_row_ptr[k][3] = static_cast<float>(normal.x());
          info_row_ptr[k][4] = static_cast<float>(normal.y());
          info_row_ptr[k][5] = static_cast<float>(normal.z());
        }
      }

      percent = static_cast<int>(100.0*(double(i) / this->scene_.nmbr_paths_));

      std::cout << "\r" << percent << "% completed (" << i << " rays): ";
      std::cout << std::string(percent/10, '@') << std::string(10 - percent/10, '=');
      std::cout.flush();

      if (cv::waitKey(1) == 13) {
        processed_rays = i+1;
        break;
      }
    }
  } else {  // Sem anti-aliasing
    for (int i = 0; i < this->scene_.nmbr_paths_; ++i) {
      // Dispara um raio n vezes em um determinado pixel.
      #pragma omp parallel for
      for (int j = 0; j < rendered_image.rows; ++j) {
        cv::Vec3d *row_ptr = rendered_image.ptr<cv::Vec3d>(j);
        GeometricInfo *info_row_ptr = geometric_information.ptr<GeometricInfo>(j);

        for (int k = 0; k < rendered_image.cols; ++k) {
          Vector3d looking_at((this->scene_.camera_.bottom_(0) + pixel_w / 2) + k*pixel_w,
                              (this->scene_.camera_.top_(1)    - pixel_h / 2) - j*pixel_h,
                              0.0);
          Vector3d direction = looking_at - this->scene_.camera_.eye_;
          direction = direction / direction.norm();

          // Perform path tracing.
          util::Ray ray(this->scene_.camera_.eye_, direction, 1);
          Vector3d additional_color = this->TracePath(ray);

          row_ptr[k][0] += additional_color(2);
          row_ptr[k][1] += additional_color(1);
          row_ptr[k][2] += additional_color(0);

          // Gather geometric information.
          util::RenderableObject *object;
          double t;
          Vector3d normal;

          this->GetNearestObjectAndIntersection(ray, &object, &t, &normal);
          Vector3d intersection_point = ray.origin + t * ray.direction;

          info_row_ptr[k][0] = static_cast<float>(intersection_point.x());
          info_row_ptr[k][1] = static_cast<float>(intersection_point.y());
          info_row_ptr[k][2] = static_cast<float>(intersection_point.z());

          info_row_ptr[k][3] = static_cast<float>(normal.x());
          info_row_ptr[k][4] = static_cast<float>(normal.y());
          info_row_ptr[k][5] = static_cast<float>(normal.z());
        }
      }

      percent = static_cast<int>(100.0 * (double(i) / this->scene_.nmbr_paths_));

      std::cout << "\r" << percent << "% completed (" << i << " rays): ";
      std::cout << std::string(percent/10, '@') << std::string(10 - percent/10, '=');
      std::cout.flush();

      if (cv::waitKey(1) == 13) {
        processed_rays = i+1;
        break;
      }
    }
  }

  std::cout << "\r100% completed (" << this->scene_.nmbr_paths_ << " rays): ";
  std::cout << std::string(10, '@') << std::endl;

  rendered_image = rendered_image / processed_rays;

  const int N = 2;
  for (int i = 0; i < N; ++i)
    cv::ximgproc::amFilter(geometric_information, rendered_image, rendered_image, 12.401, 0.8102, true);

  return rendered_image;
}

}  // namespace pt

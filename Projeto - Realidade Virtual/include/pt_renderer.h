// Copyright (c) 2015, Jubileus
//
// Project: Sucesso do verao
// Author: Rodrigo F. Figueiredo <rodrigo.figueiredo@gprt.ufpe.br>
// Creation date: 05/01/2016 (dd/mm/yyyy)

#ifndef PT_RENDERER_H
#define PT_RENDERER_H

#include <Eigen\Dense>
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>

#include <cmath>
#include <chrono>
#include <iostream>
#include <random>

#include "ray.h"
#include "renderable_object.h"
#include "sdl_object.h"

namespace pt {

class PTRenderer {
 public:
  explicit PTRenderer(const util::SDLObject &scene)
      : scene_(scene),
        generator_(std::default_random_engine(scene.random_seed_)),
        ray_generator_(std::default_random_engine(scene.random_seed_)),
        anti_aliasing_generator_(std::default_random_engine(scene.random_seed_)),
        light_generator_(std::default_random_engine(scene.random_seed_)),
        distribution_(std::uniform_real_distribution<double>(0, 1)) {}
  ~PTRenderer () {};

  cv::Mat RenderScene();

 private:
  static const double kEps;

  Eigen::Vector3d TracePath(const util::Ray &ray);
  //cv::Mat GetImageGeometricInformation();
  void GetNearestObjectAndIntersection(const util::Ray &ray,
                                       util::RenderableObject **object,
                                       double *parameter,
                                       Eigen::Vector3d *normal);
  double ScaleLightIntensity(double light_intensity,
                             const Eigen::Vector3d &light_position,
                             const util::Ray &shadow_ray);

  util::SDLObject scene_;
  std::default_random_engine generator_;
  std::default_random_engine ray_generator_;
  std::default_random_engine anti_aliasing_generator_;
  std::default_random_engine light_generator_;
  std::uniform_real_distribution<double> distribution_;
};

}  // namespace pt

#endif  // PT_RENDERER_H

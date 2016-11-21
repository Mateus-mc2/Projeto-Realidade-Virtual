// Copyright (c) 2015, Jubileus
//
// Project: Sucesso do verao
// Author: Rodrigo F. Figueiredo <rodrigo.figueiredo@gprt.ufpe.br>
// Creation date: 29/12/2015 (dd/mm/yyyy)

#ifndef SDL_OBJECT_H_
#define SDL_OBJECT_H_

#include <Eigen/Dense>

#include "camera.h"
#include "light.h"
#include "quadric.h"
#include "triangular_object.h"

#include <vector>
#include <string>

namespace util {

struct SDLObject {
  SDLObject(){}
  SDLObject(const std::string &file_name,
            const Camera &camera,
            const Eigen::Vector3d &background_color,
            double ambient_light_intensity,
            const std::vector<PointLight> &point_lights,
            const std::vector<TriangularObject> &extense_lights,
            int nmbr_paths,
            int max_depth,
            double tone_mapping,
            int random_seed,
            const std::vector<Quadric> &quadrics_objects,
            const std::vector<TriangularObject> &triangular_objects,
            int antialiasing,
            int lightsamplingtype)
      : file_name_(file_name),
        camera_(camera),
        background_color_(background_color),
        ambient_light_intensity_(ambient_light_intensity),
        point_lights_(point_lights),
        extense_lights_(extense_lights),
        nmbr_paths_(nmbr_paths),
        max_depth_(max_depth),
        tone_mapping_(tone_mapping),
        random_seed_(random_seed),
        quadrics_objects_(quadrics_objects),
        triangular_objects_(triangular_objects),
        antialiasing_(antialiasing),
        lightsamplingtype_(lightsamplingtype) {}

  ~SDLObject() {};

  std::string                   file_name_;
  Camera                        camera_;
  Eigen::Vector3d               background_color_;
  double                        ambient_light_intensity_;
  std::vector<PointLight>       point_lights_;
  std::vector<TriangularObject> extense_lights_;
  int                           nmbr_paths_;
  int                           max_depth_;
  double                        tone_mapping_;
  int                           random_seed_;
  std::vector<Quadric>          quadrics_objects_;
  std::vector<TriangularObject> triangular_objects_;
  int                           antialiasing_;
  int                           lightsamplingtype_;
};

}  // namespace util

#endif  // SDL_OBJECT_H_

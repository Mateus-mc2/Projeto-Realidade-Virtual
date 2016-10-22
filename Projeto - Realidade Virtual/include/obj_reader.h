// Copyright (c) 2015, Jubileus
//
// Project: Sucesso do verao
// Author: Rodrigo F. Figueiredo <rodrigo.figueiredo@gprt.ufpe.br>
// Creation date: 30/12/2015 (dd/mm/yyyy)

#ifndef OBJ_READER_H
#define OBJ_READER_H

#include <Eigen/Dense>

#include <vector>
#include <string>
#include <iostream>
#include <fstream>

namespace io {

class OBJReader {
 public:
  OBJReader() {};
  ~OBJReader () {};

 void ReadOBJ(const std::string &file_directory,
              const std::string &file_name,
              std::vector<Eigen::Vector3d> *vertices,  // Sera preenchido
              std::vector<Eigen::Vector3i> *faces);    // Sera preenchido

};

}  // namespace io

#endif  // OBJ_READER

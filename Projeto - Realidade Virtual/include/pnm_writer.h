// Copyright (c) 2015, Jubileus
//
// Project: Sucesso do verao
// Author: Rodrigo F. Figueiredo <rodrigo.figueiredo@gprt.ufpe.br>
// Creation date: 22/12/2015 (dd/mm/yyyy)

#ifndef PNM_WRITER_H_
#define PNM_WRITER_H_

#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>

#include <string>
#include <iostream>
#include <fstream>
#include <ctime>

namespace io {

class PNMWriter {
 public:
  PNMWriter(const std::string &base_directory) : base_directory(base_directory) {};

  void WritePNMFile(const cv::Mat &image);  // Imagem será salva no diretório 'base_directory'
                                            //  e seu nome será o horário atual.
  void WritePNMFile(const cv::Mat &image, const std::string &file_directory,
                    const std::string &file_name);


 private:
  std::string DateString() const;
  std::string base_directory; // Diretorio padrão onde serão salvas as imagens.
};

}  // namespace io

#endif  // PNM_WRITER_H_

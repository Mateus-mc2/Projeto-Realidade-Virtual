#include "pnm_writer.h"

namespace io {

void PNMWriter::WritePNMFile(const cv::Mat &image) {
  std::string file_directory = this->base_directory;
  std::string file_name      = this->DateString();

  this->WritePNMFile(image, file_directory, (file_name));
}

void PNMWriter::WritePNMFile(const cv::Mat &image, const std::string &file_directory,
                             const std::string &file_name) {
  //std::ofstream pnm_file;
  //pnm_file.open(file_directory + (file_name + ".pnm"));

  //// Cabeçalho
  //pnm_file << "P3" << std::endl;
  //pnm_file << "# " << file_name << std::endl;
  //pnm_file << image.cols << " " << image.rows << std::endl;
  //pnm_file << "255" << std::endl;

  //// Valores dos pixels
  //for (int i=0; i < image.rows; ++i) {
  //  for (int j=0; j < image.cols; ++j) {
  //    unsigned char b = image.data[i*(image.cols*3)+j*3+0];
  //    unsigned char g = image.data[i*(image.cols*3)+j*3+1];
  //    unsigned char r = image.data[i*(image.cols*3)+j*3+2];
  //    pnm_file << std::to_string(r) << std::endl;
  //    pnm_file << std::to_string(g) << std::endl;
  //    pnm_file << std::to_string(b) << std::endl;
  //  }
  //}

  //pnm_file.close();

  // Dá pra usar o OpenCV diretamente!! :'(
  std::vector<int> compression_params;
  compression_params.push_back(CV_IMWRITE_PXM_BINARY);
  compression_params.push_back(0);

  cv::imwrite(file_directory + (file_name + ".pnm"), image, compression_params);
}

std::string PNMWriter::DateString() const {
  time_t rawtime;
  tm timeinfo;
  char buffer[200];

  time(&rawtime);
  localtime_s(&timeinfo, &rawtime);

  strftime(buffer, 200, "imagem-%Y.%m.%d-%H.%M.%S", &timeinfo);
  return std::string(buffer);
}
}  // namespace io

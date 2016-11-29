// Copyright (c) 2015, Jubileus
//
// Project: Sucesso do verao
// Author: Rodrigo F. Figueiredo <rodrigo.figueiredo@gprt.ufpe.br>
// Creation date: 05/01/2016 (dd/mm/yyyy)

#include <chrono>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "gpu_path_tracer.h"
#include "gpu_sdl_reader.h"
#include "pt_renderer.h"
#include "pnm_writer.h"
#include "sdl_reader.h"

using gpu::GPUScene;
using gpu::GPUPathTracer;
using io::SDLReader;
using util::SDLObject;

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cout << "  SDL input and targets missing." << std::endl;
    return -1;
  }

  // Leitura da cena
  std::cout << "\n## Reading SDL file..." << std::endl;
  SDLReader sdl_reader;
  SDLObject sdl_object = sdl_reader.ReadSDL(argv[1], argv[2]);

  // ## Testando funcionalidade da GPU.
  GPUScene *scene = gpu::io::ReadGPUScene(argv[1], argv[2]);
  if (scene) {
    GPUPathTracer path_tracer(scene->seed);
    cv::Mat rendered_img = path_tracer.RenderScene(scene);
    cudaFree(scene);

    if (!rendered_img.empty()) cv::imshow("Rendered Image", rendered_img);
    std::cout << "  Press any key to close the image." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
  }

  cv::Mat rendered_img;

  // Processamento
  std::cout << "\n## Rendering started." << std::endl;
  pt::PTRenderer pt_renderer(sdl_object);

  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  rendered_img = pt_renderer.RenderScene();
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  std::chrono::duration<double> time_elapsed =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
  std::cout << "  Time elapsed: " << time_elapsed.count() << " seconds." << std::endl;

  cv::imshow("Rendered Image", rendered_img);
  std::cout << "  Press any key to close the image." << std::endl;
  cv::waitKey(0);
  cv::destroyAllWindows();

  //// Escrita da imagem renderizada
  rendered_img *= 255;
  std::cout << "\n## Exporting file..." << std::endl;
  io::PNMWriter pnm_mgr(argv[3]);

  if (argc > 4) {
    pnm_mgr.WritePNMFile(rendered_img, argv[3], argv[4]);
  } else {
    pnm_mgr.WritePNMFile(rendered_img);
  }

  return 0;
}

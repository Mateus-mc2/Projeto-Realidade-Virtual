// Copyright (c) 2015, Jubileus
//
// Project: Sucesso do verao
// Author: Rodrigo F. Figueiredo <rodrigo.figueiredo@gprt.ufpe.br>
// Creation date: 05/01/2016 (dd/mm/yyyy)

#include <iostream>

#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>

#include "pt_renderer.h"
#include "pnm_writer.h"
#include "sdl_reader.h"
#include "psnr.h"

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cout << "  SDL input and targets missing." << std::endl;
    return -1;
  }

  // Leitura da cena
  std::cout << "\n## Reading SDL file." << std::endl;
  io::SDLReader sdl_reader;
  util::SDLObject sdl_object = sdl_reader.ReadSDL(argv[1], argv[2]);

  // Processamento
  std::cout << "\n## Rendering started." << std::endl;
  pt::PTRenderer pt_renderer(sdl_object);

  float sigma_s[] = {16, 12.401};
  float sigma_r[] = {0.2, 0.8102};
  int N = 2; // tamanho de sigma_s/r

  for(int i = 0 ; i < N ; i++) {
    cv::Mat rendered_img = pt_renderer.RenderScene(sigma_s[i], sigma_r[i]);
  
	psnr
	// Escrita da imagem renderizada
	std::cout << "\n## Começo da exportação." << std::endl;
	io::PNMWriter pnm_mgr(argv[3]);
	if (argc > 4)
		pnm_mgr.WritePNMFile(rendered_img, argv[3], argv[4]);
	else
		pnm_mgr.WritePNMFile(rendered_img);
	}

  return 0;
}

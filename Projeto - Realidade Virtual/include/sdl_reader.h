// Copyright (c) 2015, Jubileus
//
// Project: Sucesso do verao
// Author: Rodrigo F. Figueiredo <rodrigo.figueiredo@gprt.ufpe.br>
// Creation date: 29/12/2015 (dd/mm/yyyy)

#ifndef SDL_READER_H
#define SDL_READER_H

#include "sdl_object.h"

#include <string>

namespace io {

class SDLReader {
 public:
  SDLReader() {};
  ~SDLReader () {};

  util::SDLObject ReadSDL(const std::string &file_directory, const std::string &filename) const;
};

}  // namespace io

#endif  // SDL_READER

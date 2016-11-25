#ifndef GPU_SDL_READER_H
#define GPU_SDL_READER_H

#include "gpu_scene.h"

#include <string>

namespace gpu {
namespace io {

GPUScene ReadGPUScene(const std::string &directory, const std::string &filename);

}  // namespace io
}  // namespace gpu

#endif  // GPU_SDL_READER_H_

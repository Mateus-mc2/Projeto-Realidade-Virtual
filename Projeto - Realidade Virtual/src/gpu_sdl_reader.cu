#include "gpu_sdl_reader.h"

#include <fstream>
#include <iostream>
#include <vector>

#include "gpu_camera.h"
#include "gpu_light.h"
#include "gpu_material.h"
#include "gpu_quadric.h"
#include "gpu_triangular_object.h"
#include "gpu_vector.h"

namespace gpu {
namespace io {
namespace {

void ReadOBJ(const std::string &file_directory, const std::string &file_name,
             GPUVector<float3> *vertices, GPUVector<int3> *faces) {
  std::ifstream obj_file(file_directory + file_name);
  std::string word;

  // Ler todas as linhas
  if (obj_file.is_open()) {
    while (obj_file >> word) {
      if (word[0] == '#') {                // Comentario
        std::string commentary;
        std::getline(obj_file, commentary);
      } else if (word == "v") {            // Vertice
        float3 vertex;

        obj_file >> vertex.x;
        obj_file >> vertex.y;
        obj_file >> vertex.z;

        vertices->PushBack(vertex);
      } else if (word == "f") {            // Face, composta de indices para o vetor de vertices
        int3 face;

        obj_file >> face.x;
        obj_file >> face.y;
        obj_file >> face.z;

        --face.x;
        --face.y;
        --face.z;

        faces->PushBack(face);
      } else {
        std::cout << "==obj==   BORA BOY! token nao suportado: " << word << std::endl;
        std::cout << "==obj==     Leitura interrompida." << std::endl;
        obj_file.close();
        return;  // BORA BOY! token nao suportado
      }
    }

    obj_file.close();
  }
}


}  // namespace

GPUScene* ReadGPUScene(const std::string &directory, const std::string &filename) {
  GPUCamera camera;
  float3 bg_color;

  GPUVector<GPULight> point_lights;
  GPUVector<GPUQuadric> quadric_objects;
  GPUVector<GPUTriangularObject> triangular_objects;

  bool use_anti_aliasing;

  float tone_mapping;
  float ambient_light_intensity;

  int num_paths;
  int max_depth;
  int random_seed;
  int light_sampling_type;

  std::ifstream sdl_file(directory + filename);
  std::string word;

  if (sdl_file.is_open()) {
    while (sdl_file >> word) {
      if (word[0] == '#') {                  // Comentario
        std::string commentary;
        std::getline(sdl_file, commentary);

      } else if (word == "output") {         // Nome do arquivo de saida
                                             // TODO(Mateus): we don't really use this parameter from SDL file.
        std::string output_name;
        sdl_file >> output_name;

      } else if (word == "eye") {            // Centro da camera
        float3 eye;

        sdl_file >> eye.x;
        sdl_file >> eye.y;
        sdl_file >> eye.z;

        camera.eye = eye;
      } else if (word == "ortho") {          // Viewport da camera
        float2 bottom;
        sdl_file >> bottom.x;
        sdl_file >> bottom.y;

        float2 top;
        sdl_file >> top.x;
        sdl_file >> top.y;

        camera.bottom = bottom;
        camera.top = top;
      } else if (word == "size") {           // Quantidade de pixels na horizontal e vertical
        sdl_file >> camera.width;
        sdl_file >> camera.height;

      } else if (word == "background") {     // Cor do fundo da imagem
        sdl_file >> bg_color.x;
        sdl_file >> bg_color.y;
        sdl_file >> bg_color.z;

      } else if (word == "ambient") {        // Intensidade da componente ambiente
        sdl_file >> ambient_light_intensity;

      } else if (word == "pointlight") {     // Luzes pontuais
                                             // Nao existe no formato original da especificacao
        float3 position; 
        float r, g, b, intensity;

        sdl_file >> position.x;
        sdl_file >> position.y;
        sdl_file >> position.z;
        sdl_file >> r;
        sdl_file >> g;
        sdl_file >> b;
        sdl_file >> intensity;
        
        point_lights.PushBack(gpu::GPULight(position, r, g, b, intensity));
      } else if (word == "npaths") {         // Numero de raios por pixel
        sdl_file >> num_paths;

      } else if (word == "maxdepth") {       // Quantidade maxima de reflexoes
                                             // Nao existe no formato original da especificacao
        sdl_file >> max_depth;

      } else if (word == "tonemapping") {    // Regulador de iluminacao -para pos processamento
        sdl_file >> tone_mapping;

      } else if (word == "seed") {           // Semente inteira do gerador randomico
        sdl_file >> random_seed;

      } else if (word == "objectquadric") {  // Objetos baseados em surpeficies parametricas
        float3 color;
        float a, b, c, f, g, h, p, q, r, d, refraction, ka, kd, ks, kt, n;

        sdl_file >> a;
        sdl_file >> b;
        sdl_file >> c;

        sdl_file >> f;
        sdl_file >> g;
        sdl_file >> h;

        sdl_file >> p;
        sdl_file >> q;
        sdl_file >> r;

        sdl_file >> d;

        sdl_file >> color.x;
        sdl_file >> color.y;
        sdl_file >> color.z;

        sdl_file >> refraction;
        sdl_file >> ka;
        sdl_file >> kd;
        sdl_file >> ks;
        sdl_file >> kt;
        sdl_file >> n;

        GPUMaterial new_material(color, refraction, ka, kd, ks, kt, n);
        quadric_objects.PushBack(GPUQuadric(a, b, c, f, g, h, p, q, r, d, new_material));
      } else if (word == "object") {         // Objetos baseados em malhas trianguladas
        std::string obj_file_name;
        float3 color;
        float refraction, ka, kd, ks, kt, n;

        sdl_file >> obj_file_name;

        sdl_file >> color.x;
        sdl_file >> color.y;
        sdl_file >> color.z;

        sdl_file >> refraction;
        sdl_file >> ka;
        sdl_file >> kd;
        sdl_file >> ks;
        sdl_file >> kt;
        sdl_file >> n;

        GPUMaterial new_material(color, refraction, ka, kd, ks, kt, n);
        GPUVector<float3> new_vertices;
        GPUVector<int3> new_faces;

        ReadOBJ(directory, obj_file_name, &new_vertices, &new_faces);
        int num_faces = static_cast<int>(new_faces.size());
        triangular_objects.PushBack(GPUTriangularObject(new_material, new_vertices, new_faces));
      } else if(word == "antialiasing") {
        sdl_file >> use_anti_aliasing;

      } else if(word == "lightsamplingtype") {
        sdl_file >> light_sampling_type;

      } else {
        std::cout << "  Unsupported token: " << word << std::endl;
        std::cout << "  Closing input file..." << std::endl;
        sdl_file.close();
        return nullptr;
      }
    }

    sdl_file.close();
    std::cout << "## SDL file " << filename << " successfully read ##" << std::endl;

    GPUScene *scene;
    cudaMallocManaged(&scene, sizeof(GPUScene));

    scene->camera = camera;
    scene->bg_color = bg_color;

    scene->lights = point_lights;
    scene->quadrics = quadric_objects;
    scene->triangular_objs = triangular_objects;

    scene->ambient_light_intensity = ambient_light_intensity;
    scene->tone_mapping = tone_mapping;

    scene->use_anti_aliasing = use_anti_aliasing;

    scene->num_paths = num_paths;
    scene->max_depth = max_depth;
    scene->seed = random_seed;
    scene->light_sampling_type = light_sampling_type;

    scene->max_float = std::numeric_limits<float>::max();

    return scene;
  } else {
    std::cout << "  Inexistent input file: " << word << std::endl;
    return nullptr;
  }
}

}  // namespace io
}  // namespace gpu

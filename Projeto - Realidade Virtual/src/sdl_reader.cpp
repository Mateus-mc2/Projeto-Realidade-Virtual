#include "sdl_reader.h"

#include <iostream>
#include <fstream>
#include <vector>

#include "camera.h"
#include "gpu_camera.h"
#include "gpu_light.h"
#include "gpu_material.h"
#include "gpu_quadric.h"
#include "gpu_triangular_object.h"
#include "gpu_vector.h"
#include "light.h"
#include "obj_reader.h"
#include "quadric.h"
#include "triangular_object.h"

namespace io {

util::SDLObject SDLReader::ReadSDL(const std::string &directory,
                                   const std::string &filename) const {
  // Preparar todas as variaveis que receberao os dados para a criacao do sdl_object
  std::string                         output_name;
  Eigen::Vector3f                     eye;
  Eigen::Vector2f                     bottom;
  Eigen::Vector2f                     top;
  int                                 width;
  int                                 height;
  Eigen::Vector3d                     background_color;
  double                              ambient_light_intensity;
  std::vector<util::PointLight>       point_lights;
  std::vector<util::TriangularObject> extense_lights;
  int                                 nmbr_paths;
  int                                 max_depth;
  double                              tone_mapping;
  int                                 random_seed;
  std::vector<util::Quadric>          quadrics_objects;
  std::vector<util::TriangularObject> triangular_objects;
  int                                 antialiasing;
  int                                 lightsamplingtype;

  std::ifstream sdl_file(directory + filename);
  std::string word;

  // Ler todas as linhas
  if (sdl_file.is_open()) {
    while (sdl_file >> word) {
      if (word[0] == '#') {                  // Comentario
        std::string commentary;
        std::getline(sdl_file, commentary);

      } else if (word == "output") {         // Nome do arquivo de saida
        sdl_file >> output_name;

      } else if (word == "eye") {            // Centro da camera
        float x, y, z;

        sdl_file >> x;
        sdl_file >> y;
        sdl_file >> z;

        eye[0] = x;
        eye[1] = y;
        eye[2] = z;
      } else if (word == "ortho") {          // Viewport da camera
        float bx, by, tx, ty;

        sdl_file >> bx;
        sdl_file >> by;
        sdl_file >> tx;
        sdl_file >> ty;

        bottom[0] = bx;
        bottom[1] = by;

        top[0] = tx;
        top[1] = ty;
      } else if (word == "size") {           // Quantidade de pixels na horizontal e vertical
        sdl_file >> width;
        sdl_file >> height;
      } else if (word == "background") {     // Cor do fundo da imagem
        double r, g, b;

        sdl_file >> r;
        sdl_file >> g;
        sdl_file >> b;

        background_color[0] = r;
        background_color[1] = g;
        background_color[2] = b;
      } else if (word == "ambient") {        // Intensidade da componente ambiente
        sdl_file >> ambient_light_intensity;

      } else if (word == "light") {          // Luzes extensas
        std::string obj_file_name;
        double red, green, blue, lp, light_sampling_step, light_density;

        sdl_file >> obj_file_name;
        sdl_file >> red;
        sdl_file >> green;
        sdl_file >> blue;
        sdl_file >> lp;
        sdl_file >> light_sampling_step;
        sdl_file >> light_density;

        util::Material new_material(red, green, blue, 1, 0, 0, 0, 0, 1, lp, light_sampling_step, light_density);

        std::vector<Eigen::Vector3d> new_vertices;
        std::vector<Eigen::Vector3i> new_faces;

        io::OBJReader obj_reader;
        obj_reader.ReadOBJ(directory, obj_file_name, &new_vertices, &new_faces);
        util::TriangularObject new_triangular_obj(new_material, true, new_vertices, new_faces);

        extense_lights.push_back(new_triangular_obj);
      } else if (word == "pointlight") {     // Luzes pontuais
                                             // Nao existe no formato original da especificacao
        double x, y, z, r, g, b, intensity;

        sdl_file >> x;
        sdl_file >> y;
        sdl_file >> z;
        sdl_file >> r;
        sdl_file >> g;
        sdl_file >> b;
        sdl_file >> intensity;

        Eigen::Vector3d new_position(x, y, z);
        util::PointLight new_point_light(new_position, r, g, b, intensity);

        point_lights.push_back(new_point_light);
      } else if (word == "npaths") {         // Numero de raios por pixel
        sdl_file >> nmbr_paths;

      } else if (word == "maxdepth") {       // Quantidade maxima de reflexoes
                                             // Nao existe no formato original da especificacao
        sdl_file >> max_depth;

      } else if (word == "tonemapping") {    // Regulador de iluminacao -para pos processamento
        sdl_file >> tone_mapping;

      } else if (word == "seed") {           // Semente inteira do gerador randomico
        sdl_file >> random_seed;

      } else if (word == "objectquadric") {  // Objetos baseados em surpeficies parametricas
        double a, b, c, d, e, f, g, h, j, k, red, green, blue, refraction, ka, kd, ks, kt, n;
        sdl_file >> a;
        sdl_file >> b;
        sdl_file >> c;
        sdl_file >> d;
        sdl_file >> e;
        sdl_file >> f;
        sdl_file >> g;
        sdl_file >> h;
        sdl_file >> j;
        sdl_file >> k;
        sdl_file >> red;
        sdl_file >> green;
        sdl_file >> blue;
        sdl_file >> refraction;
        sdl_file >> ka;
        sdl_file >> kd;
        sdl_file >> ks;
        sdl_file >> kt;
        sdl_file >> n;

        util::Material new_material(red, green, blue, refraction, ka, kd, ks, kt, n, 0.0, 0.0, 0.0);
        util::Quadric  new_quadric(a, b, c, d, e, f, g, h, j, k, new_material, false);

        quadrics_objects.push_back(new_quadric);
      } else if (word == "object") {         // Objetos baseados em malhas trianguladas
        std::string obj_file_name;
        double red, green, blue, refraction, ka, kd, ks, kt, n;

        sdl_file >> obj_file_name;
        sdl_file >> red;
        sdl_file >> green;
        sdl_file >> blue;
        sdl_file >> refraction;
        sdl_file >> ka;
        sdl_file >> kd;
        sdl_file >> ks;
        sdl_file >> kt;
        sdl_file >> n;

        util::Material new_material(red, green, blue, refraction, ka, kd, ks, kt, n, 0.0, 0.0, 0.0);

        std::vector<Eigen::Vector3d> new_vertices;
        std::vector<Eigen::Vector3i> new_faces;

        io::OBJReader obj_reader;
        obj_reader.ReadOBJ(directory, obj_file_name, &new_vertices, &new_faces);
        util::TriangularObject new_triangular_obj(new_material, false, new_vertices, new_faces);

        triangular_objects.push_back(new_triangular_obj);
      } else if(word == "antialiasing") {
        sdl_file >> antialiasing;

      } else if(word == "lightsamplingtype") {
        sdl_file >> lightsamplingtype;

      } else {
        std::cout << "  Unsupported token: " << word << std::endl;
        std::cout << "  Closing input file..." << std::endl;
        sdl_file.close();
        return util::SDLObject();
      }
    }

    sdl_file.close();
    std::cout << "## SDL file " << filename << " successfully read ##" << std::endl;

    util::Camera new_camera(eye, bottom, top, width, height);
    return util::SDLObject(output_name, new_camera, background_color, ambient_light_intensity,
                           point_lights, extense_lights, nmbr_paths, max_depth, tone_mapping,
                           random_seed, quadrics_objects, triangular_objects, antialiasing,
                           lightsamplingtype);
  } else {
    std::cout << "  Inexistent input file: " << word << std::endl;
    return util::SDLObject();
  }
}

gpu::GPUScene SDLReader::ReadGPUScene(const std::string &directory,
                                      const std::string &filename) const {
  gpu::GPUCamera camera;
  float3 bg_color;

  gpu::GPUVector<gpu::GPULight*> point_lights;
  gpu::GPUVector<gpu::GPUQuadric*> quadrics_objects;
  gpu::GPUVector<gpu::GPUTriangularObject*> triangular_objects;

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

        point_lights.PushBack(new gpu::GPULight(position, r, g, b, intensity));
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

        gpu::GPUMaterial new_material(color, refraction, ka, kd, ks, kt, n);
        quadrics_objects.PushBack(new gpu::GPUQuadric(a, b, c, f, g, h, p, q, r, d, new_material));
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

        gpu::GPUMaterial new_material(color, refraction, ka, kd, ks, kt, n);

        std::vector<float3> new_vertices;
        std::vector<int3> new_faces;

        io::OBJReader obj_reader;
        obj_reader.ReadOBJ(directory, obj_file_name, &new_vertices, &new_faces);
        int num_faces = static_cast<int>(new_faces.size());

        triangular_objects.PushBack(new gpu::GPUTriangularObject(new_material, new_vertices.data(),
                                                                 new_faces.data(), num_faces));
      } else if(word == "antialiasing") {
        sdl_file >> use_anti_aliasing;

      } else if(word == "lightsamplingtype") {
        sdl_file >> light_sampling_type;

      } else {
        std::cout << "  Unsupported token: " << word << std::endl;
        std::cout << "  Closing input file..." << std::endl;
        sdl_file.close();
        return gpu::GPUScene();
      }
    }

    sdl_file.close();
    std::cout << "## SDL file " << filename << " successfully read ##" << std::endl;

    return gpu::GPUScene(camera, bg_color, ambient_light_intensity, point_lights, quadrics_objects,
                         triangular_objects, use_anti_aliasing, num_paths, max_depth, tone_mapping,
                         random_seed, light_sampling_type);
  } else {
    std::cout << "  Inexistent input file: " << word << std::endl;
    return gpu::GPUScene();
  }
}

}  // namespace io

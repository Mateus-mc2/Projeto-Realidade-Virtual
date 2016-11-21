#include "sdl_reader.h"
#include "obj_reader.h"

namespace io {
util::SDLObject SDLReader::ReadSDL(const std::string &file_directory,
                                   const std::string &file_name) {
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

  std::ifstream sdl_file(file_directory + file_name);
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
        obj_reader.ReadOBJ(file_directory, obj_file_name, &new_vertices, &new_faces);
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
        obj_reader.ReadOBJ(file_directory, obj_file_name, &new_vertices, &new_faces);
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
    std::cout << "## SDL file " << file_name << " successfully read ##" << std::endl;

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

}  // namespace io

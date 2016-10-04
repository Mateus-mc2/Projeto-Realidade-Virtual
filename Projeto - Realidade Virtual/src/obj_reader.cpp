#include "obj_reader.h"

namespace io {

void OBJReader::ReadOBJ(const std::string &file_directory, const std::string &file_name,
                        std::vector<Eigen::Vector3d> &vertices,
                        std::vector<Eigen::Vector3i> &faces) {
  std::ifstream obj_file(file_directory + file_name);
  std::string word;

  // Ler todas as linhas
  if (obj_file.is_open()) {
    while (obj_file >> word) {
      if (word[0] == '#') {                // Comentario
        std::string commentary;
        std::getline(obj_file, commentary);
        std::cout << "==obj== Lido #" << commentary << std::endl;

      } else if (word == "v") {            // Vertice
        double x, y, z;
        obj_file >> x;
        obj_file >> y;
        obj_file >> z;
        Eigen::Vector3d new_vertice(x, y, z);
        vertices.push_back(new_vertice);
        std::cout << "==obj== Lido 'vertice': " << x << " " << y << " " << z << std::endl;

      } else if (word == "f") {            // Face, composta de indices para o vetor de vertices
        int idx1, idx2, idx3;
        obj_file >> idx1;
        obj_file >> idx2;
        obj_file >> idx3;
        Eigen::Vector3i new_face(idx1-1, idx2-1, idx3-1);
        faces.push_back(new_face);
        std::cout << "==obj== Lido 'face': " << idx1 << " " << idx2 << " " << idx3 << std::endl;

      } else {
        std::cout << "==obj==   BORA BOY! token nao suportado: " << word << std::endl;
        std::cout << "==obj==     Leitura interrompida." << std::endl;
        obj_file.close();
        return;  // BORA BOY! token nao suportado
      }
    }

    std::cout << "==obj== ## Arquivo OBJ "<< file_name <<" lido com sucesso" << std::endl;
    obj_file.close();
  }
}

}  // namespace io
#ifndef MATERIAL_H_
#define MATERIAL_H_

#include "util_exception.h"

namespace util {

class InvalidMaterialCoefficientsException : public UtilException {
 public:
   InvalidMaterialCoefficientsException(const std::string &error_msg)
       : UtilException(error_msg) {}
};

struct Material {
  Material () {}
  Material(double r, double g, double b, double refraction_coeff, double k_a, double k_d,
           double k_s, double k_t, int n, double lp, double light_sampling_step,
           double light_density);
  ~Material() {}

  double red;   // Componente vermelha.
  double green; // Componente verde.
  double blue;  // Componente azul.
  double refraction_coeff;  // Coeficiente de refra��o.

  // Relevantes aos objetos iluminados
  double k_a;  // Coeficiente de reflex�o ambiente.
  double k_d;  // Coeficiente de reflex�o difusa.
  double k_s;  // Coeficiente de reflex�o especular.
  double k_t;  // Coeficiente de transpar�ncia.
  int n;       // Expoente de reflex�o especular.

  // Relevantes aos objetos luminosos
  double lp;
  double light_sampling_step;
  double light_density;
};

}  // namespace util

#endif  // MATERIAL_H_

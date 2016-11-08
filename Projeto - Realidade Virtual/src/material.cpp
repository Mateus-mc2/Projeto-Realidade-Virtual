#include "material.h"

namespace util {

Material::Material(double r, double g, double b, double refraction_coeff, double k_a, double k_d,
                   double k_s, double k_t, double n, double lp, double light_sampling_step,
                   double light_density)
                   :  red(r),
                      green(g),
                      blue(b),
                      refraction_coeff(refraction_coeff),
                      k_a(k_a),
                      k_d(k_d),
                      k_s(k_s),
                      k_t(k_t),
                      n(n),
                      lp(lp),
                      light_sampling_step(light_sampling_step),
                      light_density(light_density){
  if (r < 0 || g < 0 || b < 0 || k_a < 0 || k_d < 0 || k_s < 0 || k_t < 0 || n <= 0 ||
      r > 1 || g > 1 || b > 1 || k_a > 1 || k_d > 1 || k_s > 1 || k_t > 1 || lp < 0 ||
      lp > 1 || refraction_coeff < 1) {
    throw InvalidMaterialCoefficientsException("Material coefficients out of range.");
  }
}

}  // namespace util

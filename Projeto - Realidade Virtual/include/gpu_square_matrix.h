#ifndef GPU_SQUARE_MATRIX_H_
#define GPU_SQUARE_MATRIX_H_

#include "gpu_template_matrix.h"

namespace gpu {
namespace {

__device__ float FastExponentiation(float base, int exp) {
  if (exp == 0) {
    return 1;
  } else if (exp < 0) {
    base = 1 / base;
    exp = -exp;
  }

  float result = 1.0f;

  if (exp & 1) {
    result = base;
    --exp;
  }

  while (exp) {
    base *= base;
    exp >>= 1;
  }

  result *= base;
  return result;
}

}  // namespace

template<int N>
class GPUSquareMatrixF : public GPUTemplateMatrix<float, N, N> {
 public:
  __device__ GPUSquareMatrixF() : GPUTemplateMatrix<float, N, N>() {}
  __device__ GPUSquareMatrixF(const GPUSquareMatrixD<N> &matrix)
      : GPUTemplateMatrix<float, N, N>(matrix) {}
  __device__ ~GPUSquareMatrixF() {}

  __device__ void ApplyForwardElimination(GPUSquareMatrixF<N> *upper, int *num_perm) {
    int last_pivot_col = 0;
    for (int i = 0; i < N; ++i) {
      bool found_next_pivot = false;

      for (int j = last_pivot_col; j < N && !found_next_pivot; ++j) {
        for (int k = i; k < N; ++k) {
          if ((*upper)(k, j)) {
            if (i != k) {
              upper->SwapRows(i, k);
              ++(*num_perm);
            }
            last_pivot_col = j;
            found_next_pivot = true;
          }
        }
      }

      if (found_next_pivot) {
        for (int k = i + 1; k < N; ++k) {
          float scalar_factor = (*upper)(k, last_pivot_col) / (*upper)(i, last_pivot_col);

          for (int j = last_pivot_col; j < N; ++j) {
            (*upper)(k, j) -= scalar_factor * (*upper)(i, j);
          }
        }
      }
      else {
        break;
      }
    }
  }

  __device__ float Determinant() {
    GPUSquareMatrixF<N> upper(*this);
    int num_perm = 0;
    float partial_result = 1.0f;

    ApplyForwardElimination(&upper, &num_perm);

    for (int i = 0; i < N; ++i) {
      partial_result *= upper(i, i);
    }

    return FastExponentiation(-1, num_perm) * partial_result;
  }

  __device__ GPUTemplateMatrix<float, N, 1> Solve(const GPUSquareMatrixF<N> &system,
                                                  const GPUTemplateMatrix<float, N, 1> &result) {
    GPUSquareMatrixF<N> upper(*this);
    int num_perm = 0;

    ApplyForwardElimination(&upper, &num_perm);

    return SolveUpperTriangular(upper, result);
  }

  __device__ GPUTemplateMatrix<float, N, 1> SolveUpperTriangular(
      const GPUSquareMatrixF<N> &echelon_system,
      const GPUTemplateMatrix<float, N, 1> &result) {
    GPUTemplateMatrix<float, N, 1> solution;
    ApplyForwardSubstitution(echelon_system, result, &solution);

    return solution;
  }

 private:
  __device__ void ApplyForwardSubstitution(const GPUSquareMatrixF<N> &upper,
                                           const GPUTemplateMatrix<float, N, 1> &result,
                                           GPUTemplateMatrix<float, N, 1> *solution) {
    for (int i = N - 1; i >= 0; --i) {
      if (upper(i, i) != 0) {
        float diff = 0.0f;

        for (int j = i + 1; j < N; ++j) {
          diff += upper(i, j) * (*solution)(j, 0);
        }

        (*solution)(i, 0) = (result(i, 0) - diff) / upper(i, i);
      } else {  // TODO(Mateus): find a better approach to handle systems without solution.
        solution->Fill(0.0f);
        break;
      }
    }
  }
};

typedef GPUSquareMatrixF<3> GPUSquareMatrix3F;

}  // namespace gpu

#endif  // GPU_SQUARE_MATRIX_H_

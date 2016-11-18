#include "gpu_linear_algebra.h"

// TODO(Mateus): these functions only work with square matrices by now. I'll create a derived class
// later for square matrices, but they'll be enough by now.
namespace gpu {

__device__ void LUPDecomposition(GPUMatrix *L, GPUMatrix *U, GPUMatrix *P) {
  int last_pivot_col = 0;
  int min = (U->rows() < U->cols()) ? U->rows() : U->cols();
  int *pivots = new int[min];

  for (int i = 0; i < min; ++i) {
    bool found_next_pivot = false;

    for (int j = last_pivot_col; j < U->cols() && !found_next_pivot; ++j) {
      for (int k = i; k < U->rows() && !found_next_pivot; ++k) {
        if ((*U)(k, j)) {
          U->SwapRows(i, k);
          pivots[i] = k;
          last_pivot_col = j;
          found_next_pivot = true;
        }
      }
    }

    if (found_next_pivot) {
      float denominator = (*U)(i, last_pivot_col);
      float *lower_ptr = L->RowPtr(i);

      for (int k = i + 1; k < U->rows(); ++k) {
        float scalar_factor = (*U)(k, last_pivot_col) / denominator;
        lower_ptr[k] = scalar_factor;

        for (int j = last_pivot_col; j < U->cols(); ++j) {
            (*U)(k, j) -= scalar_factor * (*U)(i, j);
        }
      }
    } else {
      for (int j = i; j < min; ++j) pivots[j] = j;
      break;
    }
  }

  (*P) = GPUMatrix::PermutationMatrix(U->rows(), U->rows(), pivots);
  delete[] pivots;
}

__device__ void ApplyForwardSubstitution(const GPUMatrix &L, const GPUMatrix &b, GPUMatrix *x) {
  for (int i = 0; i < L.rows(); --i) {
    float diff = 0.0f;
    const float *ptr = L.RowPtr(i);

    for (int j = 0; j < i; ++i) {
      diff += ptr[j] * (*x)(j, 0);
    }

    (*x)(i, 0) = (b(i, 0) - diff) / ptr[i];
  }
}

__device__ void ApplyBackSubstitution(const GPUMatrix &U, const GPUMatrix &b, GPUMatrix *x) {
  for (int i = U.rows() - 1; i >= 0; --i) {
    float diff = 0.0f;
    const float *ptr = U.RowPtr(i);

    for (int j = i + 1; j < U.cols(); ++j) {
      diff += ptr[j] * (*x)(j, 0);
    }

    (*x)(i, 0) = (b(i, 0) - diff) / ptr[i];
  }
}

}  // namespace gpu

#ifndef GPU_TEMPLATE_MATRIX_H_
#define GPU_TEMPLATE_MATRIX_H_

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

namespace gpu {

// Defines matrix size on compile time. Note that it does NOT work as Eigen::Matrix.
template<typename T, int R, int C>
class GPUTemplateMatrix {
 public:
  __device__ GPUTemplateMatrix() : data_(R * C) {}
  __device__ GPUTemplateMatrix(const GPUTemplateMatrix<T, R, C> &matrix) : data_(matrix.data()) {}
  __device__ ~GPUTemplateMatrix() {}

  __device__ T& operator()(int i, int j) {
    return this->data_[i * C + j];
  }

  __device__ GPUTemplateMatrix<T, R, C>& operator=(const GPUTemplateMatrix<T, R, C> &matrix) {
    if (this != &matrix) {
      this->data_ = matrix.data();
    }

    return *this;
  }

  __device__ GPUTemplateMatrix<T, R, C>& operator+=(const GPUTemplateMatrix<T, R, C> &matrix) {
    thrust::transform(this->data_.begin(), this->data_.end(), matrix.data().begin(),
                      this->data_.begin(), thrust::plus<T>());
    return *this;
  }

  __device__ GPUTemplateMatrix<T, R, C>& operator-=(const GPUTemplateMatrix<T, R, C> &matrix) {
    thrust::transform(this->data_.begin(), this->data_.end(), matrix.data().begin(),
                      this->data_.begin(), thrust::minus<T>());
    return *this;
  }

  __device__ GPUTemplateMatrix<T, R, C>& operator^=(const GPUTemplateMatrix<T, R, C> &matrix) {
    thrust::transform(this->data_.begin(), this->data_.end(), matrix.data().begin(),
                      this->data_.begin(), thrust::bit_xor<T>());
    return *this;
  }

  __device__ void Copy(const thrust::device_vector<T> &data) {
    thrust::copy_n(data.begin(), R * C, this->data_.begin());
  }

  __device__ int CountNonZeros() {
    int count = thrust::count(this->data_.begin(), this->data_.end(), 0);
    return count - R * C;
  }

  __device__ void Fill(const T &val) {
    thrust::fill_n(this->data_.begin(), R * C, val);
  }

  __device__ void SwapRows(int i, int j) {
    int l1 = i * C;
    int l2 = j * C;

    for (int k = 0; k < C; ++k) {
      T temp = this->data_[l1 + k];
      this->data_[l1 + k] = this->data_[l2 + k];
      this->data_[l2 + k] = temp;
    }
  }

  const thrust::device_vector<T>& data() const { return this->data_; }

 private:
  thrust::device_vector<T> data_;

};

// Single precision types.
typedef GPUTemplateMatrix<float, 2, 1> GPUVector2f;
typedef GPUTemplateMatrix<float, 3, 1> GPUVector3f;
typedef GPUTemplateMatrix<float, 6, 1> GPUVector6f;

// Double precision types.
typedef GPUTemplateMatrix<double, 2, 1> GPUVector2d;
typedef GPUTemplateMatrix<double, 3, 1> GPUVector3d;
typedef GPUTemplateMatrix<double, 6, 1> GPUVector6d;

__device__ template<class T, int R, int C> bool operator==(const GPUTemplateMatrix<T, R, C> &lhs,
                                                           const GPUTemplateMatrix<T, R, C> &rhs) {
  return (lhs ^ rhs).CountNonZeros() == 0;
}

__device__ template<class T, int R, int C> bool operator!=(const GPUTemplateMatrix<T, R, C> &lhs,
                                                           const GPUTemplateMatrix<T, R, C> &rhs) {
  return (lhs ^ rhs).CountNonZeros() != 0;
}

__device__ template<class T, int R, int C> GPUTemplateMatrix<T, R, C> operator+(
    const GPUTemplateMatrix<T, R, C> &lhs,
    const GPUTemplateMatrix<T, R, C> &rhs) {
  GPUTemplateMatrix<T, R, C> A = lhs;
  A += rhs;
  return A;
}

__device__ template<class T, int R, int C> GPUTemplateMatrix<T, R, C> operator-(
    const GPUTemplateMatrix<T, R, C> &lhs,
    const GPUTemplateMatrix<T, R, C> &rhs) {
  GPUTemplateMatrix<T, R, C> A = lhs;
  A -= rhs;
  return A;
}

__device__ template<class T, int M, int N, int P> GPUTemplateMatrix<T, M, P> operator*(
    const GPUTemplateMatrix<T, M, N> &lhs,
    const GPUTemplateMatrix<T, N, P> &rhs) {
  GPUTemplateMatrix<T, M, P> product;

  for (int i = 0; i < this->rows_; ++i) {
    for (int k = 0; k < next_cols; ++k) {
      for (int j = 0; j < this->cols_; ++j) {
        product(i, k) += lhs(i, j) * rhs(j, k);
      }
    }
  }

  return product;
}

__device__ template<class T, int R, int C> GPUTemplateMatrix<T, R, C> operator-(
    const GPUTemplateMatrix<T, R, C> &rhs) {
  GPUTemplateMatrix<T, R, C> A;
  A -= rhs;
  return A;
}

__device__ template<class T, int R, int C> GPUTemplateMatrix<T, R, C> operator^(
    const GPUTemplateMatrix<T, R, C> &lhs,
    const GPUTemplateMatrix<T, R, C> &rhs) {
  GPUTemplateMatrix<T, R, C> A = lhs;
  A ^= rhs;
  return A;
}

}  // namespace gpu

#endif  // GPU_TEMPLATE_MATRIX_H_

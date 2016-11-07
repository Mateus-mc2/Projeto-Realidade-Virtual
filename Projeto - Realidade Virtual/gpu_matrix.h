#ifndef GPU_MATRIX_H_
#define GPU_MATRIX_H_

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>

namespace gpu {

template<typename T>
class GPUMatrix {
 public:
  // TODO(Mateus): it currently needs to check dimensions right before initialization. Find a
  // better approach later.
  __device__ GPUMatrix(const thrust::device_vector<T> &data, int rows, int cols)
      : data_(data),
        rows_(rows),
        cols_(cols) {}
  __device__ GPUMatrix(const GPUMatrix<T> &M) : data_(M.data()), rows_(M.rows()), cols_(M.cols()) {}
  __device__ ~GPUMatrix() {}

  __device__ static GPUMatrix<T> Zeros(int rows, int cols) {
    thrust::device_vector<T> data(rows * cols, 0);
    return GPUMatrix<T>(data, rows, cols);
  };

  __device__ T& operator()(int row, int col) {
    return this->data_[row * this->cols_ + col];
  }

  __device__ GPUMatrix& operator=(const GPUMatrix<T> &M) {
    if (this != &M) {
      this->data_ = M.data();
      this->rows_ = M.rows();
      this->cols_ = M.cols();
    }

    return *this;
  }

  __device__ GPUMatrix<T>& operator+=(const GPUMatrix<T> &M) {
    thrust::transform(this->data_.begin(), this->data_.end(), M.data().begin(),
                      this->data_.begin(), thrust::plus<T>());
    return *this;
  }

  __device__ GPUMatrix<T>& operator-=(const GPUMatrix<T> &M) {
    thrust::transform(this->data_.begin(), this->data_.end(), M.data().begin(),
                      this->data_.begin(), thrust::minus<T>());
    return *this;
  }

  __device__ GPUMatrix<T>& operator*=(const GPUMatrix<T> &M) {
    int next_cols = M.cols();
    thrust::device_vector<T> result(this->rows_ * next_cols, 0);

    for (int i = 0; i < this->rows_; ++i) {
      for (int k = 0; k < next_cols; ++k) {
        for (int j = 0; j < this->cols_; ++j) {
          result[i * next_cols + k] += this->data_[i * this->cols_ + j] * M(j, k);
        }
      }
    }

    this->data_ = result;
    this->cols_ = next_cols;

    return *this;
  }

  __device__ GPUMatrix<T>& operator^=(const GPUMatrix<T> &M) {
    thrust::transform(this->data_.begin(), this->data_.end(), M.data().begin(),
                      this->data_.begin(), thrust::bit_xor<T>());
    return *this;
  }

  __device__ int CountNonZeros() {
    int count = thrust::count(thrust::device, this->data_.begin(),
                              this->data_.begin() + this->rows_ * this->cols_, 0);
    return (this->rows_ * this->cols_) - count;
  }

  thrust::device_vector<T> data() const { return this->data_; }
  int rows() const { return this->rows_; }
  int cols() const { return this->cols_; }

 private:
  thrust::device_vector<T> &data_;
  int rows_;
  int cols_;
};

__device__ template<class T> bool operator==(const GPUMatrix<T> &lhs, const GPUMatrix<T> &rhs) {
  return (lhs ^ rhs).CountNonZeros() == 0;
}

__device__ template<class T> bool operator!=(const GPUMatrix<T> &lhs, const GPUMatrix<T> &rhs) {
  return (lhs ^ rhs).CountNonZeros() != 0;
}

__device__ template<class T> GPUMatrix<T> operator+(const GPUMatrix<T> &lhs,
                                                    const GPUMatrix<T> &rhs) {
  GPUMatrix A = lhs;
  A += rhs;
  return A;
}

__device__ template<class T> GPUMatrix<T> operator-(const GPUMatrix<T> &lhs,
                                                    const GPUMatrix<T> &rhs) {
  GPUMatrix A = lhs;
  A -= rhs;
  return A;
}

__device__ template<class T> GPUMatrix<T> operator-(const GPUMatrix<T> &rhs) {
  GPUMatrix A = GPUMatrix::Zeros(rhs.rows(), rhs.cols());
  A -= rhs;
  return A;
}
__device__ template<class T> GPUMatrix operator*(const GPUMatrix<T> &lhs,
                                                 const GPUMatrix<T> &rhs) {
  GPUMatrix A = lhs;
  A *= rhs;
  return A;
}
__device__ template<class T> GPUMatrix operator^(const GPUMatrix<T> &lhs,
                                                 const GPUMatrix<T> &rhs) {
  GPUMatrix A = lhs;
  A ^= rhs;
  return A;
}

}  // namespace gpu

#endif  // GPU_MATRIX_H_

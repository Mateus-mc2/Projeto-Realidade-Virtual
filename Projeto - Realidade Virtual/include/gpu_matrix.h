#ifndef GPU_MATRIX_H_
#define GPU_MATRIX_H_

#include <cuda_runtime.h>

namespace gpu {

class GPUMatrix {
 public:
  __device__ GPUMatrix() : data_(nullptr), rows_(0), cols_(0) {}
  __device__ GPUMatrix(int rows, int cols)
      : data_(new float[rows * cols]),
        rows_(rows),
        cols_(cols) {}
  __device__ GPUMatrix(const GPUMatrix &matrix);
  __device__ ~GPUMatrix() { delete[] this->data_; }

  __device__ static GPUMatrix Identity(int rows, int cols);

  __device__ float& operator()(int row, int col);
  __device__ float operator()(int row, int col) const;

  __device__ GPUMatrix& operator=(const GPUMatrix &rhs);
  __device__ GPUMatrix& operator+=(const GPUMatrix &rhs);
  __device__ GPUMatrix& operator-=(const GPUMatrix &rhs);
  __device__ GPUMatrix& operator*=(const GPUMatrix &rhs);
  __device__ GPUMatrix& operator^=(const GPUMatrix &rhs);

  __device__ bool IsEmpty() const { return this->rows_ | this->cols_ == 0; }
  __device__ int CountNonZeros() const;

  // Accessors.
  __device__ const float *data() const { return this->data_; }
  __device__ int rows() const { return this->rows_; }
  __device__ int cols() const { return this->cols_; }

 private:
  __device__ void CopyFrom(const GPUMatrix &matrix);

  float *data_;
  int rows_;
  int cols_;
};

__device__ GPUMatrix operator-(const GPUMatrix &rhs);
__device__ bool operator==(const GPUMatrix &lhs, const GPUMatrix &rhs);
__device__ bool operator!=(const GPUMatrix &lhs, const GPUMatrix &rhs);
__device__ GPUMatrix operator+(const GPUMatrix &lhs, const GPUMatrix &rhs);
__device__ GPUMatrix operator-(const GPUMatrix &lhs, const GPUMatrix &rhs);
__device__ GPUMatrix operator*(const GPUMatrix &lhs, const GPUMatrix &rhs);
__device__ GPUMatrix operator^(const GPUMatrix &lhs, const GPUMatrix &rhs);

}  // namespace gpu

#endif  // GPU_MATRIX_H_

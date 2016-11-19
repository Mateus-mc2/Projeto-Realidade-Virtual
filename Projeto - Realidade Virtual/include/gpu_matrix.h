#ifndef GPU_MATRIX_H_
#define GPU_MATRIX_H_

#include <cuda_runtime.h>

namespace gpu {

class GPUMatrix {
 public:
  __host__ __device__ GPUMatrix() : data_(nullptr), rows_(0), cols_(0) {}
  __host__ __device__ GPUMatrix(int rows, int cols)
      : data_(new float[rows * cols]),
        rows_(rows),
        cols_(cols) {}
  __host__ __device__ GPUMatrix(const GPUMatrix &matrix);
  __host__ __device__ ~GPUMatrix() { delete[] this->data_; }

  __host__ __device__ static GPUMatrix Identity(int rows, int cols);
  __host__ __device__ static GPUMatrix PermutationMatrix(int rows, int cols, const int *pivots);
  __host__ __device__ static GPUMatrix Zeros(int rows, int cols);

  __host__ __device__ float& operator()(int row, int col);
  __host__ __device__ float operator()(int row, int col) const;

  __host__ __device__ GPUMatrix& operator=(const GPUMatrix &rhs);
  __host__ __device__ GPUMatrix& operator+=(const GPUMatrix &rhs);
  __host__ __device__ GPUMatrix& operator-=(const GPUMatrix &rhs);
  __host__ __device__ GPUMatrix& operator*=(const GPUMatrix &rhs);
  __host__ __device__ GPUMatrix& operator^=(const GPUMatrix &rhs);

  __host__ __device__ int CountNonZeros() const;
  __host__ __device__ bool IsEmpty() const { return this->rows_ == 0 || this->cols_ == 0; }
  __host__ __device__ float* RowPtr(int row) { return &this->data_[row * this->cols_]; }
  __host__ __device__ const float* RowPtr(int row) const { 
    return &this->data_[row * this->cols_];
  }
  __host__ __device__ void SwapRows(int i, int j);

  // Accessors.
  __host__ __device__ const float *data() const { return this->data_; }
  __host__ __device__ int rows() const { return this->rows_; }
  __host__ __device__ int cols() const { return this->cols_; }

 private:
  __host__ __device__ void CopyFrom(const GPUMatrix &matrix);

  float *data_;
  int rows_;
  int cols_;
};

__host__ __device__ GPUMatrix operator-(const GPUMatrix &rhs);
__host__ __device__ bool operator==(const GPUMatrix &lhs, const GPUMatrix &rhs);
__host__ __device__ bool operator!=(const GPUMatrix &lhs, const GPUMatrix &rhs);
__host__ __device__ GPUMatrix operator+(const GPUMatrix &lhs, const GPUMatrix &rhs);
__host__ __device__ GPUMatrix operator-(const GPUMatrix &lhs, const GPUMatrix &rhs);
__host__ __device__ GPUMatrix operator*(const GPUMatrix &lhs, const GPUMatrix &rhs);
__host__ __device__ GPUMatrix operator^(const GPUMatrix &lhs, const GPUMatrix &rhs);

}  // namespace gpu

#endif  // GPU_MATRIX_H_

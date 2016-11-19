#include "gpu_matrix.h"

namespace gpu {

__host__ __device__ GPUMatrix::GPUMatrix(const GPUMatrix &matrix)
    : data_(new float[matrix.rows() * matrix.cols()]),
      rows_(matrix.rows()),
      cols_(matrix.cols()) {
  for (int i = 0; i < this->rows_ * this->cols_; ++i) {
    this->data_[i] = matrix.data()[i];
  }
}

__host__ __device__ void GPUMatrix::CopyFrom(const GPUMatrix &matrix) {
  if (this->rows_ != matrix.rows() || this->cols_ != matrix.cols()) {
    delete[] this->data_;
    this->rows_ = matrix.rows();
    this->cols_ = matrix.cols();
    this->data_ = new float[this->rows_ * this->cols_];
  }

  for (int i = 0; i < this->rows_ * this->cols_; ++i) {
    this->data_[i] = matrix.data()[i];
  }
}

__host__ __device__ int GPUMatrix::CountNonZeros() const {
  int count = 0;

  for (int i = 0; i < this->rows_ * this->cols_; ++i) {
    if (this->data_[i] != 0) ++count;
  }

  return count;
}

__host__ __device__ GPUMatrix GPUMatrix::Identity(int rows, int cols) {
  GPUMatrix I(rows, cols);

  for (int i= 0; i < rows; ++i) {
    float *ptr = I.RowPtr(i);
    for (int j = 0; j < cols; ++j) {
      if (i == j) ptr[j] = 1;
      else ptr[j] = 0;
    }
  }

  return I;
}

__host__ __device__ GPUMatrix GPUMatrix::PermutationMatrix(int rows, int cols, const int *pivots) {
  GPUMatrix P(rows, cols);

  for (int i = 0; i < rows; ++i) {
    float *ptr = P.RowPtr(i);
    for (int j = 0; j < cols; ++j) {
      if (pivots[i] == j) ptr[j] = 1;
      else ptr[j] = 0;
    }
  }

  return P;
}

__host__ __device__ void GPUMatrix::SwapRows(int i, int j) {
  int line1 = i*this->cols_;
  int line2 = j*this->cols_;

  for (int k = 0; k < this->cols_; ++k) {
    float aux = this->data_[line1 + k];
    this->data_[line1 + k] = this->data_[line2 + k];
    this->data_[line2 + k] = aux;
  }
}

__host__ __device__ GPUMatrix GPUMatrix::Zeros(int rows, int cols) {
  GPUMatrix zeros(rows, cols);

  for (int i= 0; i < rows; ++i) {
    float *ptr = zeros.RowPtr(i);
    for (int j = 0; j < cols; ++j) ptr[j] = 0;
  }

  return zeros;
}

__host__ __device__ float& GPUMatrix::operator()(int row, int col) {
  return this->data_[row * this->cols_ + col];
}

__host__ __device__ float GPUMatrix::operator()(int row, int col) const {
  return this->data_[row * this->cols_ + col];
}

__host__ __device__ GPUMatrix& GPUMatrix::operator=(const GPUMatrix &rhs) {
  if (this != &rhs) {
    this->CopyFrom(rhs);
  }

  return *this;
}

__host__ __device__ GPUMatrix& GPUMatrix::operator+=(const GPUMatrix &rhs) {
  if (this->rows_ == rhs.rows() && this->cols_ == rhs.cols()) {
    for (int i = 0; i < this->rows_ * this->cols_; ++i) {
      this->data_[i] += rhs.data()[i];
    }
  }

  return *this;
}

__host__ __device__ GPUMatrix& GPUMatrix::operator-=(const GPUMatrix &rhs) {
  if (this->rows_ == rhs.rows() && this->cols_ == rhs.cols()) {
    for (int i = 0; i < this->rows_ * this->cols_; ++i) {
      this->data_[i] -= rhs.data()[i];
    }
  }

  return *this;
}

__host__ __device__ GPUMatrix& GPUMatrix::operator*=(const GPUMatrix &rhs) {
  if (this->cols_ == rhs.rows()) {
    GPUMatrix result(this->rows_, rhs.cols());

    for (int i = 0; i < result.rows(); ++i) {
      for (int j = 0; j < result.cols(); ++j) {
        result(i, j) = 0;

        for (int k = 0; k < this->cols_; ++k) {
          result(i, j) += this->data_[i * this->cols_ + k] * rhs(k, j);
        }
      }
    }

    this->CopyFrom(result);
  }

  return *this;
}

__host__ __device__ GPUMatrix& GPUMatrix::operator^=(const GPUMatrix &rhs) {
  if (this->rows_ == rhs.rows() && this->cols_ == rhs.cols()) {
    for (int i = 0; i < this->rows_ * this->cols_; ++i) {
      unsigned char *data_temp = reinterpret_cast<unsigned char*>(&this->data_[i]);
      const unsigned char *rhs_temp = reinterpret_cast<const unsigned char*>(&rhs.data()[i]);

      for (int j = 0; j < sizeof(float); ++j) {
        data_temp[j] ^= rhs_temp[j];
      }
    }
  }

  return *this;
}

__host__ __device__ GPUMatrix operator-(const GPUMatrix &rhs) {
  GPUMatrix result(rhs);
  for (int i = 0; i < rhs.rows(); ++i) {
    for (int j = 0; j < rhs.cols(); ++j) {
      result(i, j) = -result(i, j);
    }
  }

  return result;
}

__host__ __device__ bool operator==(const GPUMatrix &lhs, const GPUMatrix &rhs) {
  return (lhs ^ rhs).CountNonZeros() == 0;
}

__host__ __device__ bool operator!=(const GPUMatrix &lhs, const GPUMatrix &rhs) {
  return (lhs ^ rhs).CountNonZeros() != 0;
}

__host__ __device__ GPUMatrix operator+(const GPUMatrix &lhs, const GPUMatrix &rhs) {
  GPUMatrix sum(lhs);
  sum += rhs;

  return sum;
}

__host__ __device__ GPUMatrix operator-(const GPUMatrix &lhs, const GPUMatrix &rhs) {
  GPUMatrix diff(lhs);
  diff -= rhs;

  return diff;
}

__host__ __device__ GPUMatrix operator*(const GPUMatrix &lhs, const GPUMatrix &rhs) {
  GPUMatrix product(lhs.rows(), rhs.cols());

  for (int i = 0; i < product.rows(); ++i) {
    for (int j = 0; j < product.cols(); ++j) {
      product(i, j) = 0;

      for (int k = 0; k < lhs.cols(); ++k) {
        product(i, j) += lhs(i, k) * rhs(k, j);
      }
    }
  }

  return product;
}

__host__ __device__ GPUMatrix operator^(const GPUMatrix &lhs, const GPUMatrix &rhs) {
  GPUMatrix result(lhs);
  result ^= rhs;

  return result;
}

}  // namespace gpu

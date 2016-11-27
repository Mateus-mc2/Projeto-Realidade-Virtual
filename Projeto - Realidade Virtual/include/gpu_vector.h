#ifndef GPU_VECTOR_H_
#define GPU_VECTOR_H_

#include <iostream>

#include <cuda_runtime.h>

namespace gpu {

template<typename T>
class GPUVector {
 public:
  // Default initial capacity is 10.
  GPUVector() : size_(0), capacity_(10) {
    cudaMallocManaged(&this->data_, sizeof(T) * this->capacity_);
  }

  explicit GPUVector(int capacity) : size_(0), capacity_(capacity) {
    cudaMallocManaged(&this->data_, sizeof(T) * this->capacity_);
  }

  GPUVector(const GPUVector<T> &vector)
      : size_(vector.size()),
        capacity_(vector.capacity()) {
    cudaMallocManaged(&this->data_, this->capacity_);
    this->CopyFrom(vector);
  }

  ~GPUVector() { cudaFree(this->data_); }

  __host__ __device__ T& operator[](int index) { return this->data_[index]; }
  __host__ __device__ T& operator[](int index) const { return this->data_[index]; }

  GPUVector<T>& operator=(const GPUVector<T> &vector) {
    if (this != &vector) {
      this->size_ = vector.size();

      if (this->capacity_ < vector.capacity()) {
        this->capacity_ = vector.capacity();
        cudaFree(this->data_);
        cudaMallocManaged(&this->data_, sizeof(T) * this->capacity_);
      }

      this->CopyFrom(vector);
    }

    return *this;
  }

  __host__ __device__ bool IsEmpty() const { return this->size_ == 0; }
  void PushBack(const T &val) {
    if (this->size_ == this->capacity_) {
      this->ResizeVector();
    }

    this->data_[this->size_++] = val;
  }

  // Accessors.
  __host__ __device__ const T* data() const { return this->data_; }
  __host__ __device__ int size() const { return this->size_; }
  __host__ __device__ int capacity() const { return this->capacity_; }

 private:
  __host__ __device__ void CopyFrom(const GPUVector<T> &vector) {
    memcpy(this->data_, vector.data(), sizeof(T) * this->size_);
  }

  void ResizeVector() {
    this->capacity_ *= 2;
    T *new_vector;

    cudaMallocManaged(&new_vector, sizeof(T) * this->capacity_);
    memcpy(new_vector, this->data_, sizeof(T) * this->size_);
    cudaFree(this->data_);

    this->data_ = new_vector;
  }

  T *data_;
  int size_;
  int capacity_;
};

}  // namespace gpu

#endif  // GPU_VECTOR_H_

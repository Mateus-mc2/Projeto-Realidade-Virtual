#ifndef GPU_VECTOR_H_
#define GPU_VECTOR_H_

namespace gpu {

template<typename T>
class GPUVector {
public:
  // Default initial capacity is 10.
  __host__ __device__ GPUVector() : data_(new T[10]), size_(0), capacity_(10) {}
  __host__ __device__ GPUVector(const GPUStack<T> &vector)
      : data_(new T[vector.capacity()]),
        size_(vector.size()),
        capacity_(vector.capacity()) {
    this->CopyFrom(vector);
  }

  __host__ __device__ ~GPUVector() { delete[] this->data_; }

  __host__ __device__ T& operator[](int index) { return this->data_[index]; }
  __host__ __device__ T operator[](int index) const { return this->data_[index]; }

  __host__ __device__ GPUVector<T>& operator=(const GPUVector<T> &vector) {
    if (this != &vector) {
      this->size_ = vector.size();

      if (this->capacity_ < vector.capacity()) {
        delete[] this->data_;

        this->data_ = new T[vector.capacity()];
        this->capacity_ = vector.capacity();
      }

      this->CopyFrom(vector);
    }

    return *this;
  }

  __host__ __device__ bool IsEmpty() const { return this->size_ == 0; }
  __host__ __device__ void PushBack(const T &val) {
    if (this->size_ == this->capacity_) {
      this->ResizeStack();
    }

    this->data_[this->size_++] = val;
  }

  // Accessors.
  __host__ __device__ int size() const { return this->size_; }
  __host__ __device__ int capacity() const { return this->capacity_; }

private:
  __host__ __device__ void CopyFrom(const GPUVector<T> &vector) {
    for (int i = 0; i < vector.size(); ++i) {
      this->data_[i] = vector[i];
    }
  }

  __host__ __device__ void ResizeStack() {
    this->capacity_ *= 2;
    T *new_vector = new T[this->capacity_];

    for (int i = 0; i < this->size_; ++i) {
      new_vector[i] = this->data_[i];
    }

    delete[] this->data_;
    this->data_ = new_vector;
  }

  T *data_;
  int size_;
  int capacity_;
};

}

#endif  // GPU_VECTOR_H_

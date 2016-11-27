#ifndef GPU_ARRAY_H_
#define GPU_ARRAY_H_

#include <cuda_runtime.h>

namespace gpu {

template<typename T, int N>
class GPUArray{
 public:
  GPUArray() { cudaMallocManaged(&this->data_, sizeof(T) * N); }
  ~GPUArray() { 
    cudaDeviceSynchronize();
    cudaFree(this->data_);
  }

  T& operator[](int index) { return this->data_[index]; }
  T& operator[](int index) const { return this->data_[index]; }

  // Accessors.
  const T* data() const { return this->data_; }
  int size() const { return N; }
 private:
  T *data_;
};

}  // namespace gpu

#endif  // GPU_ARRAY_H_

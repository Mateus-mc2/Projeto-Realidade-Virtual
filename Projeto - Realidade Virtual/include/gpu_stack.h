#ifndef GPU_STACK_H_
#define GPU_STACK_H_

namespace gpu {

template<typename T>
class GPUStack {
 public:
  // Default initial capacity is 10.
  __host__ __device__ GPUStack() : data_(new T[10]), top_(0), capacity_(10) {}
  __host__ __device__ GPUStack(const GPUStack<T> &stack)
      : data_(new T[stack.capacity()]),
        top_(stack.top()),
        capacity_(stack.capacity()) { this->CopyFrom(stack); }
  __host__ __device__ ~GPUStack() { delete[] this->data_; }

  __host__ __device__ GPUStack<T>& operator=(const GPUStack<T> &stack) {
    if (this != &stack) {
      this->top_ = stack.top();

      if (this->capacity_ < stack.capacity()) {
        delete[] this->data_;

        this->data_ = new T[stack.capacity()];
        this->capacity_ = stack.capacity();
      }

      this->CopyFrom(stack);
    }

    return *this;
  }

  __host__ __device__ bool IsEmpty() const { return this->top_ == 0; }
  __host__ __device__ T Peek() const {
    if (this->top_ > 0) return this->data_[this->top_ - 1];
    else return T();
  }

  __host__ __device__ T Pop() {
    if (this->top_ > 0) {
      int prev_top = this->top_ - 1;
      --this->top_;

      return this->data_[prev_top];
    } else {
      return T();
    }
  }

  __host__ __device__ void Push(const T &val) {
    if (this->top_ == this->capacity_) {
      this->ResizeStack();
    }

    this->data_[this->top_++] = val;
  }

  // Accessors.
  __host__ __device__ const T* data() const { return this->data_; }
  __host__ __device__ int top() const { return this->top_; }
  __host__ __device__ int capacity() const { return this->capacity_; }

 private:
  __host__ __device__ void CopyFrom(const GPUStack<T> &stack) {
    const T *data = stack.data();

    for (int i = 0; i < stack.top(); ++i) {
      this->data_[i] = data[i];
    }
  }

  __host__ __device__ void ResizeStack() {
    this->capacity_ *= 2;
    T *new_stack = new T[this->capacity_];

    for (int i = 0; i < this->top_; ++i) {
      new_stack[i] = this->data_[i];
    }

    delete[] this->data_;
    this->data_ = new_stack;
  }

  T *data_;
  int top_;
  int capacity_;
};

}

#endif  // GPU_STACK_H_

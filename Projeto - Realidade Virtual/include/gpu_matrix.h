#ifndef GPU_MATRIX_H_
#define GPU_MATRIX_H_

#include <cusp/array1d.h>
#include <cusp/array2d.h>

namespace gpu {

// Vector types.
typedef cusp::array1d<float, cusp::device_memory> GPUVector1f;
typedef cusp::array1d<double, cusp::device_memory> GPUVector1d;

// Matrix types.
typedef cusp::array2d<float, cusp::device_memory> GPUMatrix1f;
typedef cusp::array2d<double, cusp::device_memory> GPUMatrix1d;
typedef cusp::array2d<float3, cusp::device_memory> GPUMatrix3f;
typedef cusp::array2d<double3, cusp::device_memory> GPUMatrix3d;

}  // namespace gpu

#endif  // GPU_MATRIX_H_

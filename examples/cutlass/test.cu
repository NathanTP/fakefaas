#include "cutlass/gemm/device/gemm.h"

using ColumnMajor = cutlass::layout::ColumnMajor;

using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                                ColumnMajor,  // Layout of A matrix
                                                float,        // Data-type of B matrix
                                                ColumnMajor,  // Layout of B matrix
                                                float,        // Data-type of C matrix
                                                ColumnMajor>; // Layout of C matrix

// This is a template kernel
extern "C"
template __global__ void cutlass::Kernel<CutlassGemm::GemmKernel>(CutlassGemm::GemmKernel::Params);

// This is a normal kernel
extern "C"
__global__ void InitializeMatrix_kernel(
  float *matrix,
  int rows,
  int columns,
  int seed = 0) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < rows && j < columns) {
    int offset = i + j * rows;

    // Generate arbitrary elements.
    int const k = 16807;
    int const m = 16;
    float value = float(((offset + seed) * k % m) - m / 2);

    matrix[offset] = value;
  }
}

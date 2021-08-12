// #include "cutlass/gemm/device/gemm.h"

// using ColumnMajor = cutlass::layout::ColumnMajor;

// using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
//                                                 ColumnMajor,  // Layout of A matrix
//                                                 float,        // Data-type of B matrix
//                                                 ColumnMajor,  // Layout of B matrix
//                                                 float,        // Data-type of C matrix
//                                                 ColumnMajor>; // Layout of C matrix

// // This is a template kernel
// extern "C"
// template __global__ void cutlass::Kernel<CutlassGemm::GemmKernel>(CutlassGemm::GemmKernel::Params);

// // This is a normal kernel
// extern "C"
// __global__ void InitializeMatrix_kernel(
//   float *matrix,
//   int rows,
//   int columns,
//   int seed = 0) {

//   int i = threadIdx.x + blockIdx.x * blockDim.x;
//   int j = threadIdx.y + blockIdx.y * blockDim.y;

//   if (i < rows && j < columns) {
//     int offset = i + j * rows;

//     // Generate arbitrary elements.
//     int const k = 16807;
//     int const m = 16;
//     float value = float(((offset + seed) * k % m) - m / 2);

//     matrix[offset] = value;
//   }
// }
#include <iostream>
#include <sstream>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
//#include "helper.h"
#include "cutlass/gemm/device/gemm.h"

typedef struct CudaConfig {
  dim3 grid_, block_;
  int smem_size_;
} CudaConfig;

using ColumnMajor = cutlass::layout::ColumnMajor;

using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                                ColumnMajor,  // Layout of A matrix
                                                float,        // Data-type of B matrix
                                                ColumnMajor,  // Layout of B matrix
                                                float,        // Data-type of C matrix
                                                ColumnMajor>; // Layout of C matrix

extern "C"
CutlassGemm::GemmKernel::Params *adaptSGEMMArgs(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc) {
  // Define a CUTLASS GEMM type
  CutlassGemm gemm_operator;

  CutlassGemm::Arguments args({M , N, K},  // Gemm Problem dimensions
                              {A, lda},    // Tensor-ref for source matrix A
                              {B, ldb},    // Tensor-ref for source matrix B
                              {C, ldc},    // Tensor-ref for source matrix C
                              {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                              {alpha, beta}); // Scalars used in the Epilogue

  // Launch the CUTLASS GEMM kernel.
  cudaStream_t stream = nullptr;
  gemm_operator.initialize(args, stream=stream);
  CutlassGemm::GemmKernel::Params params_ = gemm_operator.get_params();
  CutlassGemm::GemmKernel::Params *params_ptr = (CutlassGemm::GemmKernel::Params*) malloc(sizeof(params_));
  memcpy(params_ptr, &params_, sizeof(params_));
  return params_ptr;
}

extern "C"
CudaConfig *getCudaConfig(CutlassGemm::GemmKernel::Params *params_ptr) {
  CutlassGemm::ThreadblockSwizzle threadblock_swizzle;
  dim3 grid = threadblock_swizzle.get_grid_shape(params_ptr->grid_tiled_shape);
  dim3 block(CutlassGemm::GemmKernel::kThreadCount, 1, 1);
  int smem_size = int(sizeof(typename CutlassGemm::GemmKernel::SharedStorage));
  CudaConfig* ptr = new CudaConfig;
  ptr->grid_ = grid;
  ptr->block_ = block;
  ptr->smem_size_ = smem_size;
  return ptr;
}
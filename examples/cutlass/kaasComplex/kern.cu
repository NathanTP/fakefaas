#include "cutlassAdapters.h"
#include "cutlass/gemm/device/gemm.h"

using ColumnMajor = cutlass::layout::ColumnMajor;

using CutlassGemm = cutlass::gemm::device::GemmComplex<float,
                                                ColumnMajor,
                                                float,
                                                ColumnMajor,
                                                float,
                                                ColumnMajor>;

// This is a template kernel
extern "C" {
    template __global__ void cutlass::Kernel<CutlassGemm::GemmKernel>(CutlassGemm::GemmKernel::Params);
}

extern "C"
__global__ void testKernel(testStruct s) {
    printf("from kernel: anInt=%d dPtr=%p\n", s.anInt, s.dPtr);

    float sum = 0;
    for(int i = 0; i < 4096*2; i++) {
        sum += s.dPtr[i];
    }
}

// extern "C"
// __global__ void ReferenceGemm_kernel(
//   int M,
//   int N,
//   int K,
//   float alpha,
//   float const *A,
//   int lda,
//   float const *B,
//   int ldb,
//   float beta,
//   float *C,
//   int ldc) {

//   int i = threadIdx.x + blockIdx.x * blockDim.x;
//   int j = threadIdx.y + blockIdx.y * blockDim.y;

//   if (i < M && j < N) {
//     float accumulator = 0;

//     for (int k = 0; k < K; ++k) {
//       accumulator += A[i + k * lda] * B[k + j * ldb];
//     }

//     C[i + j * ldc] = alpha * accumulator + beta * C[i + j * ldc];
//   }
// }


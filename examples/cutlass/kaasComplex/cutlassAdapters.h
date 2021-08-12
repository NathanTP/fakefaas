#include <iostream>
#include <sstream>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include "cutlass/gemm/device/gemm_complex.h"

using ColumnMajor = cutlass::layout::ColumnMajor;

using RowMajor = cutlass::layout::RowMajor;

using CutlassGemm = cutlass::gemm::device::GemmComplex<float,        // Data-type of A matrix
                                                RowMajor,  // Layout of A matrix
                                                float,        // Data-type of B matrix
                                                ColumnMajor,  // Layout of B matrix
                                                float,        // Data-type of C matrix
                                                RowMajor,
                                                float>; // Layout of C matrix

typedef struct CudaConfig {
    int gridX;
    int gridY;
    int gridZ;

    int blockX;
    int blockY;
    int blockZ;

    int smem_size;
} CudaConfig;

extern "C" 
CutlassGemm::GemmKernel::Params *adaptSGEMMArgs(int M, int N, int K, 
        float alpha,
        float const *A_R, int lda, float const *A_I,
        float const *B_R, int ldb, float const *B_I,
        float beta,
        float *C_R, int ldc, float *C_I);

extern "C"
CudaConfig *getCudaConfig(int M, int N, int K);

typedef struct _testStruct {
    float *dPtr;
    int anInt;
} testStruct;

extern "C"
testStruct *getTestStruct(int anInt, float *dPtr);


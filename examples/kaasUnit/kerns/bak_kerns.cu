#include <stdint.h>
#include <stdio.h>

extern "C"
__global__ void doublifyKern(float* a)
{
    int idx = threadIdx.x + blockIdx.x*4;
    a[idx] *= 2;
}

extern "C"
__global__ void sumKern(uint32_t *input, uint32_t *out)
{
    const int tid = threadIdx.x;

    auto step_size = 1;
    int number_of_threads = blockDim.x;

    while (number_of_threads > 0)
    {
        if (tid < number_of_threads)
        {
            const auto fst = tid * step_size * 2;
            const auto snd = fst + step_size;
            input[fst] += input[snd];
        }

        step_size <<= 1; 
        number_of_threads >>= 1;
        __syncthreads();
    }

    if(tid == 0) {
        *out = input[0];
    }
}

extern "C"
__global__ void prodKern(uint64_t len, uint32_t *v0, uint32_t *v1, uint32_t *vout)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id < len) {
        vout[id] = v0[id] * v1[id];    
    }
}

// Row Major
#define flatIdx(R,C,NROW,NCOL) ((R*NCOL)+C)
// Column Major
//#define flatIdx(R,C,NROW,NCOL) ((C*NROW)+R)

// Generic matrix multiply.
// Original implementation by Aditi Singh (https://github.com/aditisingh/GPU-Gemm)
#define TILE_WIDTH 32
#define TILE_HEIGHT 32
extern "C"
__global__ void matmulKern(uint64_t *dims, float* array0, float* array1, float* outArr)
{
    uint64_t rows0 = dims[0];
    uint64_t cols0 = dims[1];
    uint64_t rows1 = dims[2];
    uint64_t cols1 = dims[3];

    //shared memory takes one tile at a time
    __shared__ float S1[TILE_WIDTH][TILE_HEIGHT];
    __shared__ float S2[TILE_HEIGHT][TILE_WIDTH];

    // Row/Column for this thread in the current block
    unsigned int tC = threadIdx.x;
    unsigned int tR = threadIdx.y;

    //row/col of output using x and y index of current thread (respectively)
    unsigned int outC = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int outR = blockIdx.x*blockDim.x + threadIdx.x;

    //register to store multiplication result initialized to zero
    float val = 0;

    //going over all tiles one by one, with each m
    for(int tileIdx = 0; tileIdx < 1 + ((rows1-1) / TILE_WIDTH); tileIdx++)
    {
        // row/col in output array for this tile 
        int tileC = tileIdx*TILE_WIDTH + tC;
        int tileR = tileIdx*TILE_WIDTH + tR;

        //copying a tile from array0
        //if the value is associated to a valid matrix coordinate in array0
        //then store it to shared memory S1
        if (outR < rows0 && tileC < rows1) {
            //storing a "valid" value from array to shared memory
            //S1[ty][tx] = array0[r + var1*rows0];
            //S1[tR][tC] = array0[outR][tileC];
            S1[tR][tC] = array0[flatIdx(outR, tileC, rows0, cols0)];
        } else {
            //storing zero, since there is no valid value
            S1[tR][tC] = 0;					
        }
        __syncthreads();

        //copying a tile from array1
        //if value is associates to a valid matrix coordinate in array1 then
        //store it to shared memory S2
        if(outC < cols1 && tileR < rows1) {
            //S2[ty][tx]=array1[var2+rows1*c];
            //S2[tR][tC] = array1[tileR][outC]
            S2[tR][tC] = array1[flatIdx(tileR, outC, rows1, cols1)];
        } else { 
            //storing zero, since no valid value
            S2[tR][tC]=0;
        }
        __syncthreads();

        //going over entire tile, ty row in S1 and tx column in S2
        for(int i=0; i < TILE_WIDTH; i++) {
            //val+=S1[ty][i]*S2[i][tx];
            val += S1[tR][i] * S2[i][tC];
        }
        __syncthreads();
    }

    //removing degenerate cases
    if(outR < rows0 && outC < cols1) {
        //saving multiplication result to global memory
        //outArr[idx]=val;	
        //outArr[outR][outC] = val;	
        outArr[flatIdx(outR, outC, rows0, cols1)] = val;
    }
}

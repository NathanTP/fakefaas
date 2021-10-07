#include <stdint.h>
#include <stdio.h>

extern "C"
__global__ void verifyAndCopy(int len, int expect, int* in, int* out)
{
    int idx = blockIdx.x *blockDim.x + threadIdx.x;
    int val = in[idx];
    if(val != expect) {
        printf("Expected %d, got %d\n", expect, val);
        assert(val == expect);
    }
    out[idx] = -val;
}

extern "C"
__global__ void verifyArr(int len, int expect, int* in)
{
    int idx = blockIdx.x *blockDim.x + threadIdx.x;
    int val = in[idx];
    if(val != expect) {
        printf("(%d) Expected %d, got %d\n", idx, expect, val);
        assert(val == expect);
    }
}

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

extern "C"
__global__ void multipleOut(uint64_t len, uint32_t *inp, uint32_t *outIncremented, uint32_t *outDoubled)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id < len) {
        outDoubled[id] = inp[id] * 2;
        outIncremented[id] = inp[id] + 1;
    }
}

// Row Major
#define flatIdx(R,C,NROW,NCOL) ((R*NCOL)+C)

// Generic matrix multiply.
// Original implementation by Aditi Singh (https://github.com/aditisingh/GPU-Gemm)
#define TILE_WIDTH 32
#define TILE_HEIGHT 32
extern "C"
__global__ void matmulKern(uint64_t *dims, float* outArr, float* array1, float* array0)
{
    uint64_t rows0 = dims[0];
    uint64_t cols0 = dims[1];
    uint64_t rows1 = dims[2];
    uint64_t cols1 = dims[3];

    //shared memory takes one tile at a time
    __shared__ float S1[TILE_WIDTH][TILE_HEIGHT];
    __shared__ float S2[TILE_HEIGHT][TILE_WIDTH];

    //threads x and y index for the current block
    unsigned int tx=threadIdx.x;	
    unsigned int ty=threadIdx.y;

    //row value using x and y index of current thread (respectively)
    unsigned int c=blockIdx.x*blockDim.x + threadIdx.x;	
    unsigned int r=blockIdx.y*blockDim.y + threadIdx.y;

    //register to store multiplication result initialized to zero
    float val=0;

    //going over all tiles one by one, with each m
    for(int m=0; m<1+((rows1-1)/TILE_WIDTH); m++)
    {
        //x and y thread value for current tile
        int var1=m*TILE_WIDTH+tx;
        int var2=m*TILE_WIDTH+ty;

        //copying a tile from array0
        //if the value is associated to a valid matrix coordinate in array0
        //then store it to shared memory S1
        if (r < rows0 && var1 < rows1) {
            //storing a "valid" value from array to shared memory
            S1[ty][tx] = array0[flatIdx(r, var1, rows0, cols0)];
        } else {
            //storing zero, since there is no valid value
            S1[ty][tx]=0;					
        }
        __syncthreads();

        //copying a tile from array1
        //if value is associates to a valid matrix coordinate in array1 then
        //store it to shared memory S2
        if(c < cols1 && var2 < rows1) {
            S2[ty][tx] = array1[flatIdx(var2, c, rows1, cols1)];
        } else { 
            //storing zero, since no valid value
            S2[ty][tx]=0;
        }
        __syncthreads();

        //going over entire tile, ty row in S1 and tx column in S2
        for(int i=0; i<TILE_WIDTH;i++) {
            val+=S1[ty][i]*S2[i][tx];
        }
        __syncthreads();
    }

    //removing degenerate cases
    if(r < rows0 && c< cols1) {
        //saving multiplication result to global memory
        outArr[flatIdx(r, c, rows0, cols1)] = val;
    }
}

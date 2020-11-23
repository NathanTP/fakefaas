#include <stdint.h>
#include <stdio.h>

__global__ void doublifyKern(float *a)
{
    int idx = threadIdx.x + blockIdx.x*4;
    a[idx] *= 2;
}

__global__ void sumKern(uint32_t* input, uint32_t *out)
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

__global__ void prodKern(uint32_t *v0, uint32_t *v1, uint32_t *vout, uint64_t *len)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id < *len) {
        vout[id] = v0[id] * v1[id];    
    }
}

extern "C"
void prod(int grid, int block, void **bufs) {
    uint32_t *v0 = (uint32_t*)bufs[0];
    uint32_t *v1 = (uint32_t*)bufs[1];
    uint64_t *len = (uint64_t*)bufs[2];
    uint32_t *vout = (uint32_t*)bufs[3];

    prodKern<<<grid, block>>>(v0, v1, vout, len);
}

extern "C"
void sum(int grid, int block, void **bufs) {
    uint32_t *in = (uint32_t*)bufs[0];
    uint32_t *out = (uint32_t*)bufs[1];

    sumKern<<<grid, block>>>(in, out);
}

extern "C"
void doublify(int grid, int block, void **bufs)
{
    float *a = (float*)bufs[0];
    doublifyKern<<<grid,block>>>(a);
}

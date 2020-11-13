#include <stdint.h>
#include <stdio.h>

__global__ void doublifyKern(float *a)
{
    int idx = threadIdx.x + threadIdx.y*4;
    a[idx] *= 2;
}

__global__ void sum(int* input)
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
    }
}

__global__ void prod(uint32_t *v0, uint32_t *v1, uint32_t *vout, uint64_t *args)
{
    int64_t len = args[0];
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id < (len / 4))
        vout[id] = v0[id] + v1[id];    
}

extern "C"
void dotprod(void **bufs, int grid, int block) {
    uint32_t *v0 = (uint32_t*)bufs[0];
    uint32_t *v1 = (uint32_t*)bufs[1];
    uint32_t *vout = (uint32_t*)bufs[2];
    uint64_t *args = (uint64_t*)bufs[3];

    prod<<<grid, block>>>(v0, v1, vout, args);
}

/* extern "C" */
/* void doublify(float *a) */
/* { */
/*     doublifyKern<<<1,4>>>(a); */
/* } */

extern "C"
void doublify(int grid, int block, void **bufs)
{
    float *a = (float*)bufs[0];
    /* float *a = (float*)bufs; */
    doublifyKern<<<grid,block>>>(a);
}

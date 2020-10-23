// Various invocation methods for libsort
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <algorithm>
#include <atomic>

#include "sort.h"
#include "utils.h"

// Perform a partial sort of bits [offset, width). boundaries will contain the
// index of the first element of each unique group value (each unique value of
// width bits), it must be 2^width elements long.
extern "C" bool gpuPartial(uint32_t* h_in, uint32_t *boundaries, size_t h_in_len, uint32_t offset, uint32_t width) {
    //auto-releases the reservation (if any) on destruction
    auto ctx = std::make_unique<cudaReservation>();
    if(!ctx->reserveDevice()) {
      return false;
    }

    int v;
    auto res = cudaDeviceGetAttribute(&v, cudaDevAttrPciBusId, 0);
    if(res != cudaSuccess) {
      fprintf(stderr, "Failed to get attr: %s\n", cudaGetErrorString(res));
      return false;
    }

    //The sort internally only supports 32bit sizes
    if(h_in_len > UINT32_MAX) {
      fprintf(stderr, "Input array length must be less than 2^32\n");
      return false;
    }
    SortState state (h_in, h_in_len);

    state.Step(offset, width);
    state.GetResult(h_in);
    state.GetBoundaries(boundaries, offset, width);

    return true;
}

// Sort provided input (h_in) in-place using the GPU
// Returns success status
extern "C" bool providedGpu(unsigned int* h_in, size_t h_in_len)
{
    //auto-releases the reservation (if any) on destruction
    auto ctx = std::make_unique<cudaReservation>();
    if(!ctx->reserveDevice()) {
      return false;
    }

    //The sort internally only supports 32bit sizes
    if(h_in_len > UINT32_MAX) {
      fprintf(stderr, "Input array length must be less than 2^32\n");
      return false;
    }
    SortState state(h_in, h_in_len);

    state.Step(0, 32);
    state.GetResult(h_in);

    return true;
}

// Sort provided input (in) using the CPU
// returns success status
extern "C" bool providedCpu(unsigned int* in, size_t len) {
    std::sort(in, in + len);
    return true;
}

extern "C" bool gpuPartialProfile(uint32_t* h_in, uint32_t *boundaries, size_t h_in_len, uint32_t offset, uint32_t width) {
  cudaProfilerStart();
  auto ret = gpuPartial(h_in, boundaries, h_in_len, offset, width);
  cudaProfilerStop();
  return ret;
}

extern "C" bool providedGpuProfile(unsigned int* h_in, size_t h_in_len) {
  cudaProfilerStart();
  auto ret = providedGpu(h_in, h_in_len);
  cudaProfilerStop();
  return ret;
}

#include <memory>
#include <atomic>

#include "utils.h"

int nCudaDevices = 0;
std::atomic_flag *cudaDeviceLocks = NULL;
semaphore *cudaDeviceAccessSemaphore;

extern "C" bool initLibSort(void)
{
  cudaError_t res;

  if(cudaDeviceLocks != NULL) {
    fprintf(stderr, "Libsort: attempted to initialize multiple times!/n"); 
    return false;
  }

  res = cudaGetDeviceCount(&nCudaDevices);
  if(res != cudaSuccess) {
    fprintf(stderr, "Failed to get device count (%d): %s\n", nCudaDevices, cudaGetErrorString(res));
    return false;
  }

  cudaDeviceLocks = new std::atomic_flag[nCudaDevices];
  for(int i = 0; i < nCudaDevices; i++) {
    cudaDeviceLocks[i].clear();
  }

  cudaDeviceAccessSemaphore = new semaphore(nCudaDevices);
  return true;
}
 
bool cudaReservation::releaseDevice(void) {
  if(deviceID >= 0) {
    cudaDeviceAccessSemaphore->up();
    cudaDeviceLocks[deviceID].clear();
  }
  return true;
}     

bool cudaReservation::reserveDevice(void) {
  cudaDeviceAccessSemaphore->down();

  // After the semaphore, at least one device is guaranteed to be available
  // (one of these CAS's will succeed)
  for(int i = 0; i < nCudaDevices; i++) {
    if(!cudaDeviceLocks[i].test_and_set()) {
      auto res = cudaSetDevice(i);
      if(res != cudaSuccess) {
        fprintf(stderr, "Failed to set current devide to %d: %s\n", i, cudaGetErrorString(res));
        return false;
      }
      deviceID = i;
      return true;
    }
  }

  fprintf(stderr, "Failed to find available device (this shouldn't happen)");
  return false;
}

// from https://en.wikipedia.org/wiki/Permuted_congruential_generator#Example_code
// using rand() is an order of magnitude slower and doesn't generate all 32bits.
#define rotr32(x, r) (x >> r | x << (-r & 31))
extern "C" void populateInput(uint32_t *arr, size_t nelem) {
	  static uint64_t       state      = 0x4d595df4d0f33173;
		static uint64_t const multiplier = 6364136223846793005u;
		static uint64_t const increment  = 1442695040888963407u;

    for(size_t i = 0; i < nelem; i++) {
		    uint64_t x = state;
        unsigned count = (unsigned)(x >> 59);

        state = x * multiplier + increment;
        x ^= x >> 18;
        arr[i] = rotr32((uint32_t)(x >> 27), count);
    }
}

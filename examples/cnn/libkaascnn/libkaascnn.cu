#include <cuda.h>
#include <stdio.h>
#include "libkaascnn.h"

extern "C" bool initLibkaascnn(void)
{
	CUresult err = cuInit(0);
	if (err != CUDA_SUCCESS) {
		fprintf(stderr, "CUDA initialisation failed with error code - %d\n", err);
		return false;
	}

  return true;
}

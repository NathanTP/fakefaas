
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
#define __shfl_sync(mask, var, lane, width) \
        __shfl((var), (lane), (width))

#define __shfl_down_sync(mask, var, offset, width) \
        __shfl_down((var), (offset), (width))

#define __shfl_up_sync(mask, var, offset, width) \
        __shfl_up((var), (offset), (width))
#endif


#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
extern "C" __global__ void fused_nn_batch_flatten_kernel0(float* __restrict__ placeholder, float* __restrict__ tensor) {
  tensor[(((int)threadIdx.x))] = placeholder[(((int)threadIdx.x))];
}

extern "C" __global__ void fused_nn_dense_nn_bias_add_nn_relu_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ placeholder2, float* __restrict__ T_relu) {
  float T_dense_rf[1];
  __shared__ float red_buf0[64];
  __shared__ float T_dense[1];
  T_dense_rf[(0)] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 2; ++k_outer) {
    T_dense_rf[(0)] = (T_dense_rf[(0)] + (placeholder[(((k_outer * 64) + ((int)threadIdx.x)))] * placeholder1[((((((int)blockIdx.x) * 128) + (k_outer * 64)) + ((int)threadIdx.x)))]));
  }
  __syncthreads();
  ((volatile float*)red_buf0)[(((int)threadIdx.x))] = T_dense_rf[(0)];
  __syncthreads();
  if (((int)threadIdx.x) < 32) {
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 32))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) < 16) {
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 16))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 8))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 4))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 2))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 1))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    T_dense[(0)] = ((volatile float*)red_buf0)[(0)];
  }
  if (((int)threadIdx.x) == 0) {
    T_relu[(((int)blockIdx.x))] = max((T_dense[(0)] + placeholder2[(((int)blockIdx.x))]), 0.000000e+00f);
  }
}

extern "C" __global__ void fused_nn_softmax_kernel0(float* __restrict__ placeholder, float* __restrict__ T_softmax_norm) {
  float normal_reduce_temp0[1];
  float red_buf0[1];
  float T_softmax_exp[1];
  float normal_reduce_temp01[1];
  float red_buf01[1];
  normal_reduce_temp0[(0)] = -3.402823e+38f;
  if (((int)threadIdx.x) < 10) {
    normal_reduce_temp0[(0)] = max(normal_reduce_temp0[(0)], placeholder[(((int)threadIdx.x))]);
  }
  uint mask[1];
  float t0[1];
  red_buf0[(0)] = normal_reduce_temp0[(0)];
  mask[(0)] = __activemask();
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 16, 32);
  red_buf0[(0)] = max(red_buf0[(0)], t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 8, 32);
  red_buf0[(0)] = max(red_buf0[(0)], t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 4, 32);
  red_buf0[(0)] = max(red_buf0[(0)], t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 2, 32);
  red_buf0[(0)] = max(red_buf0[(0)], t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 1, 32);
  red_buf0[(0)] = max(red_buf0[(0)], t0[(0)]);
  red_buf0[(0)] = __shfl_sync(mask[(0)], red_buf0[(0)], 0, 32);
  if (((int)threadIdx.x) < 10) {
    T_softmax_exp[(0)] = __expf((placeholder[(((int)threadIdx.x))] - red_buf0[(0)]));
  }
  normal_reduce_temp01[(0)] = 0.000000e+00f;
  if (((int)threadIdx.x) < 10) {
    normal_reduce_temp01[(0)] = (normal_reduce_temp01[(0)] + __shfl_sync(__activemask(), T_softmax_exp[(0)], ((int)threadIdx.x), 32));
  }
  uint mask1[1];
  float t01[1];
  red_buf01[(0)] = normal_reduce_temp01[(0)];
  mask1[(0)] = __activemask();
  t01[(0)] = __shfl_down_sync(mask1[(0)], red_buf01[(0)], 16, 32);
  red_buf01[(0)] = (red_buf01[(0)] + t01[(0)]);
  t01[(0)] = __shfl_down_sync(mask1[(0)], red_buf01[(0)], 8, 32);
  red_buf01[(0)] = (red_buf01[(0)] + t01[(0)]);
  t01[(0)] = __shfl_down_sync(mask1[(0)], red_buf01[(0)], 4, 32);
  red_buf01[(0)] = (red_buf01[(0)] + t01[(0)]);
  t01[(0)] = __shfl_down_sync(mask1[(0)], red_buf01[(0)], 2, 32);
  red_buf01[(0)] = (red_buf01[(0)] + t01[(0)]);
  t01[(0)] = __shfl_down_sync(mask1[(0)], red_buf01[(0)], 1, 32);
  red_buf01[(0)] = (red_buf01[(0)] + t01[(0)]);
  red_buf01[(0)] = __shfl_sync(mask1[(0)], red_buf01[(0)], 0, 32);
  if (((int)threadIdx.x) < 10) {
    T_softmax_norm[(((int)threadIdx.x))] = (__shfl_sync(__activemask(), T_softmax_exp[(0)], ((int)threadIdx.x), 32) / red_buf01[(0)]);
  }
}

extern "C" __global__ void fused_nn_dense_nn_bias_add_nn_relu_1_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ placeholder2, float* __restrict__ T_relu) {
  float T_dense_rf[1];
  __shared__ float red_buf0[64];
  __shared__ float T_dense[1];
  T_dense_rf[(0)] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 13; ++k_outer) {
    if (((k_outer * 64) + ((int)threadIdx.x)) < 784) {
      T_dense_rf[(0)] = (T_dense_rf[(0)] + (placeholder[(((k_outer * 64) + ((int)threadIdx.x)))] * placeholder1[((((((int)blockIdx.x) * 784) + (k_outer * 64)) + ((int)threadIdx.x)))]));
    }
  }
  __syncthreads();
  ((volatile float*)red_buf0)[(((int)threadIdx.x))] = T_dense_rf[(0)];
  __syncthreads();
  if (((int)threadIdx.x) < 32) {
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 32))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) < 16) {
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 16))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 8))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 4))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 2))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 1))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    T_dense[(0)] = ((volatile float*)red_buf0)[(0)];
  }
  if (((int)threadIdx.x) == 0) {
    T_relu[(((int)blockIdx.x))] = max((T_dense[(0)] + placeholder2[(((int)blockIdx.x))]), 0.000000e+00f);
  }
}

extern "C" __global__ void fused_nn_dense_nn_bias_add_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ placeholder2, float* __restrict__ T_add) {
  float T_dense_rf[1];
  __shared__ float red_buf0[64];
  __shared__ float T_dense[1];
  T_dense_rf[(0)] = 0.000000e+00f;
  T_dense_rf[(0)] = (T_dense_rf[(0)] + (placeholder[(((int)threadIdx.x))] * placeholder1[(((((int)blockIdx.x) * 64) + ((int)threadIdx.x)))]));
  __syncthreads();
  ((volatile float*)red_buf0)[(((int)threadIdx.x))] = T_dense_rf[(0)];
  __syncthreads();
  if (((int)threadIdx.x) < 32) {
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 32))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) < 16) {
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 16))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 8))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 4))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 2))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 1))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    T_dense[(0)] = ((volatile float*)red_buf0)[(0)];
  }
  if (((int)threadIdx.x) == 0) {
    T_add[(((int)blockIdx.x))] = (T_dense[(0)] + placeholder2[(((int)blockIdx.x))]);
  }
}


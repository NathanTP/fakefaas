import pycuda.driver as cuda
import pycuda.autoinit 
from pycuda.compiler import SourceModule
import numpy as np

from mnist import MNIST
import pathlib


import sys
from tvm import relay
from tvm.relay import testing
import tvm
from tvm import te
from tvm.contrib import graph_runtime


'''Manual parser that uses PyCuda to run TVM's MNIST model. Intended to assist debugging. '''


def main():
    index = 0

    image, label = readData(index)
    params = loadParams()

    #load the library onto the gpu
    mod = loadLibrary()

    #0. null op
    n0 = loadData(image)
    
    tmp = np.zeros(shape=(1, 784), dtype=np.float32)

    n1 = loadData(tmp)

    #1. fused_nn_batch_flatten
    func = mod.get_function("fused_nn_batch_flatten_kernel0")
    func(n1, n0, block=(784, 1, 1))
    
    #2. null op (f1)
    n2 = loadData(params['p0'])

    ''' Example of how to print out the results of a step. '''
    #tmp = np.zeros(shape=(128, 784), dtype=np.float32) 
    #cuda.memcpy_dtoh(tmp, n2)
    #print(tmp)


    #3. null op (f2)
    #n3 = loadData(params['fc1_bias'])
    n3 = loadData(params['p1'])


    #4. first relu 
    tmp = np.zeros(shape=(128), dtype=np.float32)
    n4 = loadData(tmp)

    func = mod.get_function("fused_nn_dense_nn_bias_add_nn_relu_1_kernel0")
    func(n1, n2, n4, n3, block=(64, 1, 1), grid=(128, 1))


    #5. null op (p2)
    n5 = loadData(params['p2'])

    #6. null op (p3)
    n6 = loadData(params['p3'])

    #7. fused_nn_dense_nn_bias_add_nn_relu"
    tmp = np.zeros(shape=(1, 64), dtype=np.float32)
    n7 = loadData(tmp)

    func = mod.get_function("fused_nn_dense_nn_bias_add_nn_relu_kernel0")
    func(n4, n5, n7, n6, block=(64, 1, 1), grid=(64, 1))

    #8. null op (p4)
    n8 = loadData(params['p4'])

    #9. null op (p5)
    n9 = loadData(params['p5'])

    #10. fused_nn_dense_nn_bias_add"
    tmp = np.zeros(shape=(1, 10), dtype=np.float32) 
    n10 = loadData(tmp)

    func = mod.get_function("fused_nn_dense_nn_bias_add_kernel0")
    func(n7, n8, n10, n9, block=(64, 1, 1), grid=(10, 1))

    cuda.memcpy_dtoh(tmp, n10)
    
    #11. fused_nn_softmax"
    tmp = np.zeros(shape=(1, 10), dtype=np.float32)
    n11 = loadData(tmp)

    func = mod.get_function("fused_nn_softmax_kernel0")
    func(n10, n11, block=(32, 1, 1), grid=(10, 1))

    cuda.memcpy_dtoh(tmp, n11)
    print(tmp)



'''Load data onto the gpu. Assumes it is a numpy array, or that it can be converted into one with .asnumpy().'''
def loadData(data):
    if (not 'numpy' in str(type(data))):
        data = convertArray(data)
    a_gpu = cuda.mem_alloc(data.nbytes)
    cuda.memcpy_htod(a_gpu, data)
    return a_gpu


def convertArray(arr):
    return arr.asnumpy()


def processImage(img):
    new_img = np.zeros(shape=(1, 1, 28, 28), dtype=np.float32)
    for i in range(28):
        for j in range(28):
            new_img[0][0][i][j] = img[i * 28 + j]/255

    return new_img


def readData(index):
    mnistData, images, labels = loadMnist()
    return processImage(images[index]), labels[index]


''' Loads MNIST data. '''
def loadMnist(path=pathlib.Path("fakedata").resolve(), dataset='test'):
	mnistData = MNIST(str(path))
	
	if dataset == 'train':
		images, labels = mnistData.load_training()
	else:
		images, labels = mnistData.load_testing()

	images = np.asarray(images).astype(np.float32)
	labels = np.asarray(labels).astype(np.uint32)
	return mnistData, images, labels


''' Launches TVM to get the parameters. '''
def loadParams():
    batch_size = 1
    num_class = 10
    image_shape = (1, 28, 28)

    mod, params = relay.testing.mlp.get_workload(batch_size)

    target = tvm.target.cuda()
    with tvm.transform.PassContext():
        graphMod = relay.build(mod, target, params=params)

    return graphMod.get_params()
    

def loadLibrary():
    mod = SourceModule("""
        


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

extern "C" __global__ void fused_nn_batch_flatten_kernel0(float* __restrict__ tensor, float* __restrict__ placeholder) {
  tensor[(((int)threadIdx.x))] = placeholder[(((int)threadIdx.x))];
}

extern "C" __global__ void fused_nn_dense_nn_bias_add_nn_relu_1_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
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

extern "C" __global__ void fused_nn_dense_nn_bias_add_nn_relu_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
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

extern "C" __global__ void fused_nn_dense_nn_bias_add_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ placeholder2) {
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


        """)


    return mod







main()












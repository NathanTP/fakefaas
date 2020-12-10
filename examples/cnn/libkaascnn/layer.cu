#include "layer.h"
#include "cuda.h"
#include <iostream>
#include <fstream>
#include <exception>

//XXX
extern "C" void printLayerWeights(layerParams_t *l)
{
  float *t;
  if(l->onDevice) {
    t = (float*)malloc(l->N * l->M * sizeof(float));
    cudaMemcpy(t, l->weight, l->N * l->M * sizeof(float), cudaMemcpyDeviceToHost);
  } else {
    t = l->weight;
  }
  fprintf(stderr,"Ready to print\n");
  for(int i = 0; i < l->N; i++) {
    for(int j = 0; j < l->M; j++) {
      printf("P%d,%d: %f\n", i, j, t[i + j*l->N]);
    }
  }
}

//XXX
extern "C" void printLayerOutputs(layerParams_t *l)
{
  float *t;
  if(l->onDevice) {
    t = (float*)malloc(l->O * sizeof(float));
    cudaMemcpy(t, l->output, l->O * sizeof(float), cudaMemcpyDeviceToHost);
  } else {
    t = l->output;
  }

  for(int i = 0; i < l->O; i++) {
      printf("P%d: %f\n", i, t[i]);
  }
}

void clearLayer(layerParams_t *l)
{
  if(l->onDevice) {
    cudaMemset(l->output, 0x00, sizeof(float) * l->O);
    cudaMemset(l->preact, 0x00, sizeof(float) * l->O);
  } else {
    memset(l->output, 0x00, sizeof(float) * l->O);
    memset(l->preact, 0x00, sizeof(float) * l->O);
  }
}

// Move layerParams to/from the device. If l is already on the desired memory,
// these just return p.
extern "C" layerParams_t *layerParamsToHost(layerParams_t *p)
{
  if(p->onDevice == false) {
    return p;
  }

  layerParams_t *hostL = (layerParams_t *)malloc(sizeof(layerParams_t));

  hostL->M = p->M;
  hostL->N = p->N;
  hostL->O = p->O;
  hostL->onDevice = false;

  hostL->bias = (float*)malloc(sizeof(float)* p->N);  
  hostL->weight = (float*)malloc(sizeof(float)* p->N * p->M);  

  hostL->output = NULL;
  hostL->preact = NULL;

  cudaMemcpy(hostL->bias, p->bias, sizeof(float)*p->N, cudaMemcpyDeviceToHost);
  cudaMemcpy(hostL->weight, p->weight, sizeof(float)* p->N * p->M, cudaMemcpyDeviceToHost);

  return hostL;
}

extern "C" layerParams_t *layerParamsToDevice(layerParams_t *p)
{
  if(p->onDevice == true) {
    return p;
  }

  layerParams_t *devL = (layerParams_t*)malloc(sizeof(layerParams_t));

  devL->M = p->M;
  devL->N = p->N;
  devL->O = p->O;
  devL->onDevice = true;

	cudaMalloc(&devL->bias, sizeof(float) * p->N);
	cudaMalloc(&devL->weight, sizeof(float) * p->M * p->N);

	cudaMalloc(&devL->output, sizeof(float) * p->O);
	cudaMalloc(&devL->preact, sizeof(float) * p->O);

  cudaMemcpy(devL->bias, p->bias, sizeof(float)*p->N, cudaMemcpyHostToDevice);
  cudaMemcpy(devL->weight, p->weight, sizeof(float)* p->N * p->M, cudaMemcpyHostToDevice);

  return devL;
}

// Layer param intializers. The default is just random data (suitable for training).
extern "C" layerParams_t *defaultLayerParams(int M, int N, int O)
{
  layerParams_t *p = (layerParams_t*)malloc(sizeof(layerParams_t));

  p->M = M;
  p->N = N;
  p->O = O;
  p->onDevice = false;

  p->bias = (float*)malloc(sizeof(float)*N);
  p->weight = (float*)malloc(sizeof(float)*N*M);
  
  p->output = NULL;
  p->preact = NULL;

	for (int i = 0; i < N; ++i) {
	  p->bias[i] = 0.5f - float(rand()) / float(RAND_MAX);

		for (int j = 0; j < M; ++j) {
			(p->weight)[i + (j*N)] = 0.5f - float(rand()) / float(RAND_MAX);
		}
	}

  return p;
}

extern "C" layerParams_t *layerParamsFromFile(char *path)
{
  layerParams_t *p = (layerParams_t*)malloc(sizeof(layerParams_t));
  p->onDevice = false;

  std::ifstream inF(path, std::ios::binary);
  if(inF.fail()) {
    fprintf(stderr, "Failed to open layer file: %s\n", path);
    return NULL;
  }

  inF.read((char*)&p->M, sizeof(int));
  inF.read((char*)&p->N, sizeof(int));
  inF.read((char*)&p->O, sizeof(int));
  if(inF.fail()) {
    fprintf(stderr, "Failed to read from layer file: %s\n", path);
    return NULL;
  }

  p->bias = (float*)malloc(sizeof(float) * p->N);  
  p->weight = (float*)malloc(sizeof(float)* p->N * p->M);  

  p->output = NULL;
  p->preact = NULL;

  inF.read((char*)p->bias, sizeof(float) * p->N);
  inF.read((char*)p->weight, sizeof(float) * p->M * p->N);
  if(inF.fail()) {
    fprintf(stderr, "Failed to read from layer file: %s\n", path);
    return NULL;
  }

  inF.close();

  return p;
}

// Save parameters to a file
extern "C" int layerParamsToFile(layerParams_t *p, char *path)
{
  int ret = 1;

  std::ofstream outF(path, std::ios::binary | std::ios::trunc);
  if(outF.fail()) {
    printf("Failed to write layer to %s\n", path);
    return 0;
  }

  layerParams_t *hostParams = layerParamsToHost(p);
  outF.write((char*)&hostParams->M, sizeof(int));
  outF.write((char*)&hostParams->N, sizeof(int));
  outF.write((char*)&hostParams->O, sizeof(int));
  outF.write((char*)hostParams->bias, sizeof(float) * hostParams->N);
  outF.write((char*)hostParams->weight, sizeof(float) * hostParams->M * hostParams->N);
  outF.close();

  if(outF.fail()) {
    printf("Failed to write layer to %s\n", path);
    ret = 0;
  }

  if(p->onDevice) {
    freeLayerParams(hostParams);
    free(hostParams);
  }

  return ret;
}

// Frees the device buffers within l, doesn't free l itself
extern "C" void freeLayerParams(layerParams_t *p)
{
  if(p->onDevice) {
    cudaFree(p->output);
    cudaFree(p->preact);
    cudaFree(p->bias);
    cudaFree(p->weight);
  } else {
    free(p->bias);
    free(p->weight);
  }
}

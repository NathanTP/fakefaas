#ifndef _LAYER_H_
#define _LAYER_H_

#include <stdbool.h>

typedef struct {
  // These are the persistent parameters of the layer, they are either host or
  // device pointers depending on onDevice.
  float *bias;
  float *weight;

  // These are temporary device buffers for a layer. They are null when
  // onDevice==false
  float *output;
  float *preact;

  int M;
  int N;
  int O;
  bool onDevice;
} layerParams_t;

#ifdef __cplusplus
//XXX
extern "C" void printLayerWeights(layerParams_t *l);
extern "C" void printLayerOutputs(layerParams_t *l);

// Move layerParams to/from the device. If l is already on the desired memory,
// these just return p.
extern "C" layerParams_t *layerParamsToHost(layerParams_t *p);
extern "C" layerParams_t *layerParamsToDevice(layerParams_t *p);

// Layer param intializers. These return layers in host memory, you must load
// them onto the device using LayerParamsToDevice(). The default is just random
// data (suitable for training).
extern "C" layerParams_t *defaultLayerParams(int M, int N, int O);
extern "C" layerParams_t *layerParamsFromFile(char *path);

void clearLayer(layerParams_t *l);

// Save parameters to a file
extern "C" int layerParamsToFile(layerParams_t *p, char *path);

// Frees the device buffers within l, doesn't free l itself
extern "C" void freeLayerParams(layerParams_t *p);
#else
//XXX
void printLayerWeights(layerParams_t *l);
void printLayerOutputs(layerParams_t *l);

layerParams_t *layerParamsToHost(layerParams_t *p);
layerParams_t *layerParamsToDevice(layerParams_t *p);
layerParams_t *defaultLayerParams(int M, int N, int O);
layerParams_t *layerParamsFromFile(char *path);
int layerParamsToFile(char *path);
void freeLayerParams(layerParams_t *p);
#endif

#endif

#include <cstdlib>
#include <vector>
#include <memory>

#ifndef LAYER_H
#define LAYER_H

class Layer {
	public:
	int M, N, O;

  // All of the following float* are device pointers
	
  // Temporary storage for forward pass
  float *output;
	float *preact;

  // These define the model at this layer
	float *bias;
	float *weight;

  // Temporary storage for backprop
	float *d_output;
	float *d_preact;
	float *d_weight;

	Layer(int M, int N, int O);
  Layer(std::string path, bool enableTrain);

	~Layer();

  bool save(std::string path);
	void setOutput(float *data);
	void clear();
	void bp_clear();
};

#endif

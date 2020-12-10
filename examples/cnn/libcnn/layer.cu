#include "layer.h"
#include "util.h"
#include <iostream>
#include <fstream>
#include <string>
#include <exception>

// Constructor
Layer::Layer(int M, int N, int O)
{
	this->M = M;
	this->N = N;
	this->O = O;

	float h_bias[N];
	float h_weight[N][M];

	output = NULL;
	preact = NULL;
	bias   = NULL;
	weight = NULL;

	for (int i = 0; i < N; ++i) {
		h_bias[i] = 0.5f - float(rand()) / float(RAND_MAX);

		for (int j = 0; j < M; ++j) {
			h_weight[i][j] = 0.5f - float(rand()) / float(RAND_MAX);
		}
	}

	cudaMalloc(&output, sizeof(float) * O);
	cudaMalloc(&preact, sizeof(float) * O);

	cudaMalloc(&bias, sizeof(float) * N);

	cudaMalloc(&weight, sizeof(float) * M * N);

	cudaMalloc(&d_output, sizeof(float) * O);
	cudaMalloc(&d_preact, sizeof(float) * O);
	cudaMalloc(&d_weight, sizeof(float) * M * N);

	cudaMemcpy(bias, h_bias, sizeof(float) * N, cudaMemcpyHostToDevice);

	cudaMemcpy(weight, h_weight, sizeof(float) * M * N, cudaMemcpyHostToDevice);
}

// Load from a previously saved layer. Path should point to a file as saved by
// Layer::export(). If enableTrain is true, memory will be allocated for
// training purposes, otherwise the layer is loaded for inference only and can
// only be used for forward_pass().
Layer::Layer(std::string path, bool enableTrain)
{
  std::ifstream inF(path, std::ios::binary);
  if(inF.fail()) {
    throw std::runtime_error("Failed to open layer file\n");
  }

  inF.read((char*)&M, sizeof(int));
  inF.read((char*)&N, sizeof(int));
  inF.read((char*)&O, sizeof(int));
  if(inF.fail()) {
    throw std::runtime_error("Failed to read layer file\n");
  }

  float h_bias[N];
  float h_weight[N][M];

  inF.read((char*)h_bias, sizeof(float) * N);
  inF.read((char*)h_weight, sizeof(float) * M * N);
  inF.close();

	cudaMalloc(&output, sizeof(float) * O);
	cudaMalloc(&preact, sizeof(float) * O);

	cudaMalloc(&bias, sizeof(float) * N);
	cudaMalloc(&weight, sizeof(float) * M * N);

	cudaMemcpy(bias, h_bias, sizeof(float) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(weight, h_weight, sizeof(float) * M * N, cudaMemcpyHostToDevice);

  if(enableTrain) {
    cudaMalloc(&d_output, sizeof(float) * O);
    cudaMalloc(&d_preact, sizeof(float) * O);
    cudaMalloc(&d_weight, sizeof(float) * M * N);
  }
}

// Destructor
Layer::~Layer()
{
	cudaFree(output);
	cudaFree(preact);

	cudaFree(bias);

	cudaFree(weight);

	cudaFree(d_output);
	cudaFree(d_preact);
	cudaFree(d_weight);
}

// Save the layer to a file
bool Layer::save(std::string path) {
  std::ofstream outF(path, std::ios::binary | std::ios::trunc);
  if(outF.fail()) {
    printf("Failed to write layer to %s\n", path.c_str());
    return false;
  }

  float h_bias[N];
  float h_weight[N][M];

  cudaMemcpy(h_bias, bias, sizeof(float) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_weight, weight, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

  outF.write((char*)&M, sizeof(int));
  outF.write((char*)&N, sizeof(int));
  outF.write((char*)&O, sizeof(int));
  outF.write((char*)h_bias, sizeof(float) * N);
  outF.write((char*)h_weight, sizeof(float) * M * N);
  outF.close();

  if(outF.fail()) {
    printf("Failed to write layer to %s\n", path.c_str());
    return false;
  }

  return true;
}

// Send data one row from dataset to the GPU
void Layer::setOutput(float *data)
{
	cudaMemcpy(output, data, sizeof(float) * O, cudaMemcpyHostToDevice);
}

// Reset GPU memory between iterations
void Layer::clear()
{
	cudaMemset(output, 0x00, sizeof(float) * O);
	cudaMemset(preact, 0x00, sizeof(float) * O);
}

void Layer::bp_clear()
{
	cudaMemset(d_output, 0x00, sizeof(float) * O);
	cudaMemset(d_preact, 0x00, sizeof(float) * O);
	cudaMemset(d_weight, 0x00, sizeof(float) * M * N);
}


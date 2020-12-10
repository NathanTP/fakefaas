#include "layer.h"
#include <string>

#ifndef MODEL_H
#define MODEL_H

class Model {
  public:
  Layer *l_input;
  Layer *l_c1;
  Layer *l_s1;
  Layer *l_f;

  Model();
  Model(std::string modelDir, bool enableTrain);
  ~Model();


  // Returns a prediction for the digit image (0-9)
  unsigned int Classify(float data[28][28]);

  // Internal forward pass, all state remains on the GPU. You probably want to
  // run Classify() if all you care about is the prediction.
  void ForwardPass(float data[28][28]);

  // Calculates the error of the previous forward pass with the training label
  // targetLabel and prepares the model for back propagation.
  float BackPassPrepare(unsigned char targetLabel);

  // Perform back propagation
  void BackPass(void);

  // Serialize the model to a directory
  bool save(std::string modelDir);

  private:
  // this is actually a cublasHandle_t* but I don't want clients to have to include cuda headers
  void *blas;
};

#endif

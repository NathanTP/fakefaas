#include "model.h"
#include "util.h"
#include <sys/stat.h>
#include <string.h>
#include <cublas_v2.h>

Model::Model(void)
{
  blas = malloc(sizeof(cublasHandle_t));
	cublasCreate((cublasHandle_t*)blas);

  l_input = new Layer(0, 0, 28*28);
  l_c1 = new Layer(5*5, 6, 24*24*6);
  l_s1 = new Layer(4*4, 1, 6*6*6);
  l_f = new Layer(6*6*6, 10, 10);
}

Model::Model(std::string modelDir, bool enableTrain)
{
  blas = malloc(sizeof(cublasHandle_t));
	cublasCreate((cublasHandle_t*)blas);

  l_input = new Layer(modelDir + "/l_input", enableTrain);
  l_c1 = new Layer(modelDir + "/l_c1", enableTrain);
  l_s1 = new Layer(modelDir + "/l_s1", enableTrain);
  l_f = new Layer(modelDir + "/l_f", enableTrain);
}

Model::~Model()
{
  delete l_input;
  delete l_c1;
  delete l_s1;
  delete l_f;
}

// Returns a prediction for the digit image (0-9)
unsigned int Model::Classify(float data[28][28])
{
	float res[10];

	ForwardPass(data);

	unsigned int max = 0;

	cudaMemcpy(res, l_f->output, sizeof(float) * 10, cudaMemcpyDeviceToHost);

	for (int i = 1; i < 10; ++i) {
		if (res[max] < res[i]) {
			max = i;
		}
	}

	return max;
}

void Model::ForwardPass(float data[28][28])
{
	l_input->clear();
	l_c1->clear();
	l_s1->clear();
	l_f->clear();

	l_input->setOutput((float *)data);
	
	fp_preact_c1<<<64, 64>>>((float (*)[28])l_input->output, (float (*)[24][24])l_c1->preact, (float (*)[5][5])l_c1->weight);
	fp_bias_c1<<<64, 64>>>((float (*)[24][24])l_c1->preact, l_c1->bias);
	apply_step_function<<<64, 64>>>(l_c1->preact, l_c1->output, l_c1->O);

	fp_preact_s1<<<64, 64>>>((float (*)[24][24])l_c1->output, (float (*)[6][6])l_s1->preact, (float (*)[4][4])l_s1->weight);
	fp_bias_s1<<<64, 64>>>((float (*)[6][6])l_s1->preact, l_s1->bias);
	apply_step_function<<<64, 64>>>(l_s1->preact, l_s1->output, l_s1->O);

	fp_preact_f<<<64, 64>>>((float (*)[6][6])l_s1->output, l_f->preact, (float (*)[6][6][6])l_f->weight);
	fp_bias_f<<<64, 64>>>(l_f->preact, l_f->bias);
	apply_step_function<<<64, 64>>>(l_f->preact, l_f->output, l_f->O);
}

float Model::BackPassPrepare(unsigned char targetLabel)
{
  float err;

  l_f->bp_clear();
  l_s1->bp_clear();
  l_c1->bp_clear();

  // Euclid distance of train_set[i]
  makeError<<<10, 1>>>(l_f->d_preact, l_f->output, targetLabel, 10);
  cublasSnrm2(*(cublasHandle_t*)blas, 10, l_f->d_preact, 1, &err);

  return err;
}

void Model::BackPass(void)
{
	bp_weight_f<<<64, 64>>>((float (*)[6][6][6])l_f->d_weight, l_f->d_preact, (float (*)[6][6])l_s1->output);
	bp_bias_f<<<64, 64>>>(l_f->bias, l_f->d_preact);

	bp_output_s1<<<64, 64>>>((float (*)[6][6])l_s1->d_output, (float (*)[6][6][6])l_f->weight, l_f->d_preact);
	bp_preact_s1<<<64, 64>>>((float (*)[6][6])l_s1->d_preact, (float (*)[6][6])l_s1->d_output, (float (*)[6][6])l_s1->preact);
	bp_weight_s1<<<64, 64>>>((float (*)[4][4])l_s1->d_weight, (float (*)[6][6])l_s1->d_preact, (float (*)[24][24])l_c1->output);
	bp_bias_s1<<<64, 64>>>(l_s1->bias, (float (*)[6][6])l_s1->d_preact);

	bp_output_c1<<<64, 64>>>((float (*)[24][24])l_c1->d_output, (float (*)[4][4])l_s1->weight, (float (*)[6][6])l_s1->d_preact);
	bp_preact_c1<<<64, 64>>>((float (*)[24][24])l_c1->d_preact, (float (*)[24][24])l_c1->d_output, (float (*)[24][24])l_c1->preact);
	bp_weight_c1<<<64, 64>>>((float (*)[5][5])l_c1->d_weight, (float (*)[24][24])l_c1->d_preact, (float (*)[28])l_input->output);
	bp_bias_c1<<<64, 64>>>(l_c1->bias, (float (*)[24][24])l_c1->d_preact);

	apply_grad<<<64, 64>>>(l_f->weight, l_f->d_weight, l_f->M * l_f->N);
	apply_grad<<<64, 64>>>(l_s1->weight, l_s1->d_weight, l_s1->M * l_s1->N);
	apply_grad<<<64, 64>>>(l_c1->weight, l_c1->d_weight, l_c1->M * l_c1->N);
}

bool Model::save(std::string modelDir)
{
  errno = 0;
  if(mkdir(modelDir.c_str(), 0700 ) != 0) {
    if(errno != EEXIST) {
      printf("Failed to save model: %s", strerror(errno));
      return false;
    }
  }


  if(!l_input->save(modelDir + "/l_input")) {
    printf("Failed to save model\n");
    return false;
  }

  if(!l_c1->save(modelDir + "/l_c1")) {
    printf("Failed to save model\n");
    return false;
  }

  if(!l_s1->save(modelDir + "/l_s1")) {
    printf("Failed to save model\n");
    return false;
  }

  if(!l_f->save(modelDir + "/l_f")) {
    printf("Failed to save model\n");
    return false;
  }

  return true;
}

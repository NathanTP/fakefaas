#include "model.h"
#include "kernels.h"
#include "cuda.h"

/*=====================================================
 * High level helpers (used by FaaS+GPU configurations)
 *=====================================================
 */
extern "C" modelState_t *newModel(layerParams_t *input, layerParams_t *c1, layerParams_t *s1, layerParams_t *fin)
{
  modelState_t *m = (modelState_t*)malloc(sizeof(modelState_t));
  m->input = input;
  m->c = c1;
  m->s = s1;
  m->fin = fin;
  return m;
}

extern "C" bool classifyFull(modelState_t *m, float inp[28][28], float output[10])
{
  clearLayer(m->input);
  clearLayer(m->c);
  clearLayer(m->s);
  clearLayer(m->fin);

  cudaMemcpy(m->input->output, inp, 28*28*sizeof(float), cudaMemcpyHostToDevice);

  /* Real KaaS requires a generic signature */
  void* cArgs[] {(void*)m->input->output, (void*)m->c->weight, (void*)m->c->bias, (void*)m->c->preact, (void*)m->c->output};
  void* sArgs[] {(void*)m->c->output, (void*)m->s->weight, (void*)m->s->bias, (void*)m->s->preact, (void*)m->s->output};
  void* fArgs[] {(void*)m->s->output, (void*)m->fin->weight, (void*)m->fin->bias, (void*)m->fin->preact, (void*)m->fin->output};

  kaasLayerCForward(64, 64, cArgs);
  kaasLayerSForward(64, 64, sArgs);
  kaasLayerFForward(64, 64, fArgs);

  cudaMemcpy(output, m->fin->output, 10*sizeof(float), cudaMemcpyDeviceToHost);

  return true;
}

extern "C" int classify(modelState_t *m, float inp[28][28])
{
  float res[10];  

  if(!classifyFull(m, inp, res)) {
      return -1;
  }

  int max = 0;

  for (int i = 0; i < 10; ++i) {
    if (res[max] < res[i]) {
        max = i;
    }
  }

  return max;
}

/*=====================================================
 * Lowest level interface (used by KaaS)
 * The dimensions of the NN are hard-coded here. In theory we could pass them
 * as additional arguments but it's not worth the complication for now.
 *=====================================================
*/
// Input is the image
extern "C" void kaasLayerCForward(int grid, int block, void **bufs)
{
    float *input  = (float*)bufs[0];
    float *weight = (float*)bufs[1];
    float *bias   = (float*)bufs[2];
    float *preact = (float*)bufs[3];
    float *output = (float*)bufs[4];

	fp_preact_c1<<<grid, block>>>((float (*)[28])input, (float (*)[5][5])weight, (float (*)[24][24])preact);
	fp_bias_c1<<<grid, block>>>(bias, (float (*)[24][24])preact);
	apply_step_function<<<grid, block>>>(24*24*6, preact, output);
}

// Intermediate layer takes output of layerC
extern "C" void kaasLayerSForward(int grid, int block, void **bufs)
{
    float *input  = (float*)bufs[0];
    float *weight = (float*)bufs[1];
    float *bias   = (float*)bufs[2];
    float *preact = (float*)bufs[3];
    float *output = (float*)bufs[4];

	fp_preact_s1<<<grid, block>>>((float (*)[24][24])input, (float (*)[4][4])weight, (float (*)[6][6])preact);
	fp_bias_s1<<<grid, block>>>(bias, (float (*)[6][6])preact);
	apply_step_function<<<grid, block>>>(6*6*6, preact, output);
}

// Output is the predictions (array of 9 floats with the probability estimates
// for each digit, take the max for the prediction)
extern "C" void kaasLayerFForward(int grid, int block, void **bufs)
{
    float *input  = (float*)bufs[0];
    float *weight = (float*)bufs[1];
    float *bias   = (float*)bufs[2];
    float *preact = (float*)bufs[3];
    float *output = (float*)bufs[4];

	fp_preact_f<<<grid, block>>>((float (*)[6][6])input, (float (*)[6][6][6])weight, preact);
	fp_bias_f<<<grid, block>>>(bias, preact);
	apply_step_function<<<grid, block>>>(10, preact, output);
}

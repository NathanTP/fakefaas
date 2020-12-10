#define USE_MNIST_LOADER
#define MNIST_FLOAT
#include "mnist.h"
#include "libcnn.h"

#include <iostream>
#include <fstream>
#include <string>

#include <cstdio>
#include <time.h>

const static float threshold = 1.0E-02f;

static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

// Train the model
static void learn(Model *m);

// Evaluate a trained model, returns the error rate
static void test(Model *m);

static inline bool loaddata()
{
  int err;
	err = mnist_load("../data/train-images-idx3-ubyte", "../data/train-labels-idx1-ubyte",
		&train_set, &train_cnt);
  if(err != 0) {
    fprintf(stderr, "Failed to load training data\n");
    return false;
  }
	err = mnist_load("../data/t10k-images-idx3-ubyte", "../data/t10k-labels-idx1-ubyte",
		&test_set, &test_cnt);
  if(err != 0) {
    fprintf(stderr, "Failed to load t10k-images data\n");
    return false;
  }
  return true;
}

int main(int argc, const  char **argv)
{
  Model *m;
  if(argc > 1) {
    m = new  Model(std::string(argv[1]), true); 
  } else {
    m = new Model();
  }
	srand(time(NULL));

  bool ok = initLibfaascnn();
  if(!ok) {
    return 1;
  }

  ok = loaddata();
  if(!ok) {
    return 1;
  }

  if(argc > 1) {
    test(m);
  } else {
    learn(m);
    m->save("testModel");
    test(m);
  }

  delete m;
	return 0;
}

static void learn(Model *m)
{
	float err;
	int iter = 50;

	fprintf(stdout ,"Learning\n");

	while (iter < 0 || iter-- > 0) {
		err = 0.0f;

    printf("Running iteration for %d inputs\n", train_cnt);
		for (unsigned int i = 0; i < train_cnt; ++i) {
			m->ForwardPass(train_set[i].data);
			err += m->BackPassPrepare(train_set[i].label);
			m->BackPass();
		}

		err /= train_cnt;
		fprintf(stdout, "error: %e\n", err);

		if (err < threshold) {
			fprintf(stdout, "Training complete, error less than threshold\n\n");
			break;
		}
	}
}

// Perform forward propagation of test data
static void test(Model *m)
{
	int error = 0;

	// for (unsigned int i = 0; i < test_cnt; ++i) {
	for (unsigned int i = 0; i < test_cnt; ++i) {
		if (m->Classify(test_set[i].data) != test_set[i].label) {
			++error;
		}
	}

	fprintf(stdout, "Error Rate: %.2lf%%\n",
		double(error) / double(test_cnt) * 100.0);
}

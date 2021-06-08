# CNN on CUDA
A hand-written CNN in CUDA for MNIST. This serves as a reference for how a NN
could be decomposed into fakefaas. Obviously, more realistic networks would not
be handwritten like this, but we would expect a framework to generate something
similar.

# Setup
You'll need to get the mnist dataset first. If you're on the agpu machines (Berkeley internal), you can get these from /nscratch/datasets/mnist. On RISE machines this is "/work/datasets/mnist". Otherwise we expect the following files in data/:

    * t10k-images-idx3-ubyte
    * t10k-labels-idx1-ubyte
    * train-images-idx3-ubyte
    * train-labels-idx1-ubyte

# Acknowledgments
Code adapted from https://github.com/catchchaos/CUDA-CNN

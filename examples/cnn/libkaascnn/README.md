# Kernel-as-a-Service Compatible CNN
This is an MNIST cnn that has been factored to have more explicit inputs and
outputs, and a clear separation between host and device code. It's intended to
be a demonstration of how a real KaaS system could work. In reality, it's
unlikely that humans would write code like this, instead we'd expect it to be
targeted by compilers, ML frameworks, and DSLs.

This version only supports inference (for now), but there is also a libcnn
(../libcnn) that is more standalone but supports training too. The input to
libkaascnn should be a pre-trained model from ../testLibcnn (see ../testCKaas).

## Acknowledgments
Code adapted from https://github.com/catchchaos/CUDA-CNN

# Attribution
This code taken from: https://github.com/NVIDIA/cuda-samples:3342d60
Lightly adapted by Nathan Pemberton



# Quick Start

Run `make` to compile `jacobi.ptx`, and then you can run `python test.py` to see the result.
Make sure you have nvcc == 11.4!

# Explanation

Jacobi method tries to solve A*x = b.
Jacobi kernel takes in five inputs: N(numRows), A, b, x(output when iters is odd), x_new(output when iters is even), d(error)
Our python test takes two inputs: N(numRows), iters(max iterations)
In each iteration, x and x_new is switched, so that's why there are two separate calls for the kernel.

## Further Reference
Check `python/test/kaas/kaasTest.py` to see a kaas version test for the Jacobi kernel.

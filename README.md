# FakeFaaS
FakeFaaS is an experimental platform for prototyping some faas-related things,
especially GPU support (including kernel as a service). The code is provided as
a python package under python/ which also includes unit tests. The examples
directory contains various experiments and example applications.

## Quickstart
You can install the python package. I'll assume you are actively developing the
project so we'll install it in development mode so you don't have to re-install
any time you change the code:

    cd python
    python setup.py develop

Next you should make sure everything works:

    cd tests
    ./runAll.py

You can now start using stuff in examples/.

## GPU Support
Some of the functionality in this project requires a GPU, but not all. In
particular, libff.kaas requires CUDA support, as does the libff.invoke remote
functions with enableGpu=True. If you have nvcc (CUDA) installed, then the
package will include these features.  Otherwise you will be limited to a
subset. 

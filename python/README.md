# FakeFaaS
FakeFaaS is an experimental platform for prototyping some faas-related things,
especially GPU support (including kernel as a service).

## Quickstart
You can install the python package. I'll assume you are actively developing the
project so we'll install it in development mode so you don't have to re-install
any time you change the code:

    python setup.py develop

Next you should make sure everything works:

    cd tests
    ./runAll.sh

## GPU Support
Some of the functionality in this project requires a GPU, but not all. In
particular, libff.kaas requires CUDA support, as does the libff.invoke remote
functions with enableGpu=True. If you have nvcc (CUDA) installed, then the
package will include these features.  Otherwise you will be limited to a
subset. 

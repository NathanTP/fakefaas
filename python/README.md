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

### Install Anna
We also need to install anna (a type of kv database similar to redis), here are the steps:
1. Use `git clone git@github.com:hydro-project/anna.git` to clone the repo.
2. Before building anna, we check the prerequisites by running `git submodule init; git submodule update` and `./common/scripts/install-dependencies(-osx).sh`.
3. Then, build anna by running `./scripts/build.sh`.
4. You can check if anna is successfully installed by running `./scripts/run-tests.sh`.
5. Then, we need to set up anna client in python. We do this by running `python(3) setup.py install` in the `anna/client/python` directory.
6. Finally, for testing purposes, we need to set up a environment variable for the absolute path of anna directory. You can do this by `export Anna='{absolute directory of anna}'`. Notice that it is temporary only for your current shell. You can search more about how to add it permanently online.

## GPU Support
Some of the functionality in this project requires a GPU, but not all. In
particular, libff.kaas requires CUDA support, as does the libff.invoke remote
functions with enableGpu=True. If you have nvcc (CUDA) installed, then the
package will include these features.  Otherwise you will be limited to a
subset. 

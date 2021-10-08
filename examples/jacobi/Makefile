70FOUND:=$(shell lspci | grep "V100")

ifneq ($(70FOUND), )
	COMPUTE_CAP:=70
else
	COMPUTE_CAP:=35
endif

NVCC := /usr/local/cuda-11.4/bin/nvcc
NVCC_OPTS := -I./Common -O3 -g -arch=compute_$(COMPUTE_CAP) -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -fPIC -m64 -std=c++14 -rdc=true
LIBS :=  -lcuda

all: jacobi.cubin

jacobi.cubin: jacobi.cu
	/usr/local/cuda-11.4/bin/nvcc \
	-I./Common \
	-O0 -DNDEBUG -Xcompiler=-fPIC \
	--cubin \
	-Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing \
	-gencode=arch=compute_$(COMPUTE_CAP),code=sm_$(COMPUTE_CAP) -std=c++11 \
	-rdc=true \
	-x cu -c $^ -o $@
# $(NVCC) $^ -I./Common -rdc=true --cubin -gencode arch=compute_$(COMPUTE_CAP),code=sm_$(COMPUTE_CAP) -O3 -g -Xcompiler -Wall -Xcompiler -Wextra -std=c++14 -o $@

# main.o: main.cpp
# 	$(NVCC) $^ --shared $(NVCC_OPTS) $(LIBS) -o $@

# jacobiCudaGraphs: jacobi.cubin main.o
# 	nvcc $^ $(NVCC_OPTS) $(LIBS) -x cu -o $@

clean:
	rm -f *.cubin *.o

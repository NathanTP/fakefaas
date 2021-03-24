NVCC := nvcc
NVCC_OPTS := -O3 -g -arch=compute_35 -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -fPIC -m64 -std=c++14
LIBS :=  -lcuda

all: libkaascnn.cubin libkaascnn.so


libkaascnn.cubin: kernels.cu
	nvcc $^ --cubin -gencode arch=compute_35,code=sm_35 -O3 -g -Xcompiler -Wall -Xcompiler -Wextra -std=c++14 -o $@

libkaascnn.so: layer.o kernels.o model.o libkaascnn.o
	$(NVCC) $^ --shared $(NVCC_OPTS) $(LIBS) -o $@

%.o: %.cu
	$(NVCC) -c $< $(NVCC_OPTS)

clean:
	rm -f libkaascnn.so *.o

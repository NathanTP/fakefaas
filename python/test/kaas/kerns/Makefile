NVCC := nvcc
NVCC_OPTS := -O3 -g -arch=compute_35 -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -fPIC -m64 -std=c++14
LIBS :=  -lcuda

libkaasMicro.cubin: kerns.cu
	nvcc $^ --cubin -gencode arch=compute_35,code=sm_35 -O3 -g -Xcompiler -Wall -Xcompiler -Wextra -std=c++14 -o $@

libkaasMicro.so: kerns.o
	$(NVCC) $^ --shared $(NVCC_OPTS) $(LIBS) -o $@

%.o: %.cu
	$(NVCC) -c $< $(NVCC_OPTS)

clean:
	rm -f *.cubin *.o

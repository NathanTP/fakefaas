NVCC := nvcc
NVCC_OPTS := -O3 -g -arch=compute_35 -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -fPIC -m64 -std=c++14
LIBS :=  -lcuda -lcublas
# NVCC_OPTS := -O0 -g -std=c++14 -lcuda -lcublas

libcnn.so: layer.o model.o util.o
	$(NVCC) $^ --shared $(NVCC_OPTS) $(LIBS) -o $@

%.o: %.cu
	$(NVCC) -c $< $(NVCC_OPTS)

clean:
	rm -f libcnn.so *.o

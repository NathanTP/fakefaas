70FOUND:=$(shell lspci | grep "V100")

ifneq ($(70FOUND), )
	COMPUTE_CAP:=70
else
	COMPUTE_CAP:=35
endif


NVCC := nvcc
NVCC_OPTS := -O3 -g -arch=compute_$(COMPUTE_CAP) -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -fPIC -m64 -std=c++14
LIBS :=  -lcuda -lcublas
# NVCC_OPTS := -O0 -g -std=c++14 -lcuda -lcublas

libcnn.so: layer.o model.o util.o
	$(NVCC) $^ --shared $(NVCC_OPTS) $(LIBS) -o $@

%.o: %.cu
	$(NVCC) -c $< $(NVCC_OPTS)

clean:
	rm -f libcnn.so *.o

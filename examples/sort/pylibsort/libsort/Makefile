70FOUND:=$(shell lspci | grep "V100")

ifneq ($(70FOUND), )
	COMPUTE_CAP:=70
else
	COMPUTE_CAP:=35
endif


NVCC := nvcc
NVCC_OPTS := -O3 -g -arch=compute_$(COMPUTE_CAP) -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -fPIC -m64 -std=c++14

libsort.so: sort.o scan.o invokers.o utils.o
	$(NVCC) --shared $(NVCC_OPTS) $^ -o $@

%.o: %.cu
	$(NVCC) -c $< $(NVCC_OPTS) $(INCLUDES)

clean:
	rm -f *.o *.so 

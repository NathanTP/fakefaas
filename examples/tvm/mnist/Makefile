NVCC := nvcc
NVCC_OPTS := -O3 -g -arch=compute_35 -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -fPIC -m64 -std=c++14
LIBS :=  -lcuda
PYTHON := python
INDEX := 0

.DEFAULT_GOAL = run

code.cubin: mod.cu
	nvcc $^ --cubin -gencode arch=compute_35,code=sm_35 -O3 -g -Xcompiler -Wall -Xcompiler -Wextra -std=c++14 -o $@

%.o: %.cu
	$(NVCC) -c $< $(NVCC_OPTS) 

run: code.cubin  
	./getData.sh
	${PYTHON} tvm_mnist.py ${INDEX}
	${PYTHON} parserUtils.py
    

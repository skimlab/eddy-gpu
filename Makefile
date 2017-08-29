BIN               := eddy-gpu 

NVCC ?= nvcc
INCD = -I/opt/apps/intel14/boost/1.51.0/include -I"./"
# NVCCFLAGS := --ptxas-options=-v -O3 -G -g 
NVCCFLAGS := -O3 


# files
CU_SOURCES        := $(wildcard k*.cu)
C_ROUTINES        := $(wildcard r*.cu)
C_SOURCES        := $(wildcard m*.cu)
HEADERS           := $(wildcard *.h)
CU_OBJS           := $(patsubst %.cu, %.obj, $(CU_SOURCES))
CROUTINE_OBJS           := $(patsubst %.cu, %.o, $(C_ROUTINES))
C_OBJS           := $(patsubst %.cu, %.o, $(C_SOURCES))

%.obj : %.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

%.o : %.cu
	$(NVCC) $(NVCCFLAGS) -c $(INCD) -o $@ $<

$(BIN): $(CU_OBJS) $(CROUTINE_OBJS) $(C_OBJS)
	$(NVCC) -o $(BIN) $(CU_OBJS) $(CROUTINE_OBJS) $(C_OBJS) $(INCD)

clean:
	rm -f $(BIN) *.o *.obj

#-----------------------------------
CHECK_PARAMS = -DCHECK
DTYPE   = -DFP32 
#-----------------------------------

CC       =  gcc
CFLAGS   = -fopenmp 
CLINKER  =  gcc

#-------------------------------------------------------------------
#BLIS OPTFLAGS
BLIS_FLAGS = -fomit-frame-pointer -O3 -mavx2 -mfma -mfpmath=sse -march=znver1 -mno-avx256-split-unaligned-store -Wall -Wno-unused-function -Wfatal-errors -fPIC -std=c99 -D_POSIX_C_SOURCE=200112L
OPTFLAGS = $(BLIS_FLAGS) -D$(MICRO) -DMR=$(MR) -DNR=$(NR) -DKR=$(KR)
#-------------------------------------------------------------------
#OPTIMUM FLAGS <WITHOUT MEM ALLIGN>
OPTFLAGS = -O3  -Wall -Wno-unused-function -Wfatal-errors -fPIC -ftree-vectorize -fverbose-asm -std=c99 -D$(MICRO) -DMR=$(MR) -DNR=$(NR) -DKR=$(KR)
#-------------------------------------------------------------------
#OPTIMUM FLAGS <MEM ALLIGN>
#OPTFLAGS = -O3 -mavx2 -mfma -mfpmath=sse -funsafe-math-optimizations -ffp-contract=fast -mno-avx256-split-unaligned-store -march=znver1 -Wall -Wno-unused-function -Wfatal-errors -fPIC -ftree-vectorize -fverbose-asm -std=c11 -D$(MICRO) -DMR=$(MR) -DNR=$(NR) -DKR=$(KR)

# ALIGN ALLOC: -std=c11
#BLIS: -std=c99

INCLUDE  =
LIBS     =
ifeq ($(MICRO), MK_BLIS)
	INCLUDE := -I/mnt/beegfs/colab/el2mapeh/software/blis/install/include/
	LIBS    := -lblis -L/mnt/beegfs/colab/el2mapeh/software/blis/install/lib/
endif


TEST_GEMM = test_gemm.x

default: $(TEST_GEMM)

test_gemm.x: test_gemm.o sutils.o gemm_blis.o inutils.o gemm_block.o  neon/gemm_blis_neon_fp32.o modelLevel/model_level.o 
	$(CLINKER) $(OPTFLAGS) -o test_gemm.x test_gemm.o gemm_block.o model_level.o sutils.o gemm_blis.o inutils.o gemm_blis_neon_fp32.o ../libgemm.a -lm -fopenmp $(LIBS)

.c.o:
	$(CC) $(CFLAGS) $(OPTFLAGS) $(CHECK_PARAMS) $(DTYPE) -c $*.c $(INCLUDE) $(LIBS)


clean:
	rm *.o $(TEST_GEMM) 



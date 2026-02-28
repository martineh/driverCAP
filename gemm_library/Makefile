
CC       := gcc
ARCH     := -march=native
FLAGS    := -O3 -fopenmp -Wall -Wextra -fPIC -std=c99 ${ARCH}
LIBS     := -lm
INCLUDES := -I.

vpath %.c ./
vpath %.h ./

OBJDIR=build

all: gemm_library

install: all
	install -m 644 libgemm.a ../lib/
	install -m 644 libgemm.so ../lib/

gemm_library: 
	${CC} ${FLAGS} -c gemm_library.c gemm_block.c ${LIBS} $(INCLUDES)
	${CC} ${FLAGS} -shared -o libgemm.so gemm_library.o gemm_block.o ${LIBS}
	ar rcs libgemm.a gemm_block.o gemm_library.o 

clean:
	rm libgemm.a
	rm libgemm.so
	rm *.o

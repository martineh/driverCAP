
#include "dtypes.h"

void gemm_block( size_t, size_t, size_t, float, float *, size_t, float *, size_t, float, float *, size_t,
			   float *, float *, size_t, size_t, size_t);

    void gemm_blis_B3A2C0( char, char, char, char, char, size_t, size_t, size_t, float, DTYPE *, size_t, DTYPE *, size_t, DTYPE, DTYPE *, size_t,
			   DTYPE *, DTYPE *, size_t, size_t, size_t);

void gemm_base_Cresident( char, int, int, int, DTYPE, DTYPE *, int, DTYPE *, int, DTYPE, DTYPE *, int );
void gemm_base_ABresident( char, char, int, int, int, DTYPE, DTYPE *, int, DTYPE *, int, DTYPE, DTYPE *, int );

void pack_RB( char, char, int, int, DTYPE *, int, DTYPE *, int );
void pack_CB( char, char, int, int, DTYPE *, int, DTYPE *, int );
void unpack_RB( char, char, int, int, DTYPE *, int, DTYPE *, int );
void unpack_CB( char, char, int, int, DTYPE *, int, DTYPE *, int );

//-----------------------------------------------------------------------------------
void pack_RB_v( char, char, int, int, DTYPE *, int, DTYPE *, int );
void pack_CB_v( char, char, int, int, DTYPE *, int, DTYPE *, int );
//-----------------------------------------------------------------------------------



#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

#include "../qblis.h"

//-----------------------------------------------------------------------------------------------------------------------
// [*] 2v = 2v * 8x(fp32) = 16 
void gemm_microkernel_Cresident_AMD_avx256_2vx6_fp32( char, int, int, int, float, float *, float *, float, float *, int );
void gemm_microkernel_Cresident_AMD_avx256_6x2v_fp32(char, int, int, int, float, float *, float *, float, float *, int ); 
void gemm_microkernel_Cresident_AMD_avx256_6x2v_nopack_fp32(char, int, int, int, float, float *, int, float *, float, float *, int ); 
void gemm_microkernel_Cresident_AMD_avx256_6x2v_nopack_unroll_fp32(char, int, int, int, float, float *, int, float *, float, float *, int ); 
//-----------------------------------------------------------------------------------------------------------------------
// [*] 3v = 3v * 8x(fp32) = 24 
void gemm_microkernel_Cresident_AMD_avx256_3vx4_fp32( char, int, int, int, float, float *, float *, float, float *, int );
void gemm_microkernel_Cresident_AMD_avx256_4x3v_fp32( char, int, int, int, float, float *, float *, float, float *, int );
void gemm_microkernel_Cresident_AMD_avx256_4x3v_nopack_fp32(char, int, int, int, float, float *, int, float *, float, float *, int );
//-----------------------------------------------------------------------------------------------------------------------

void gemm_microkernel_Cresident_AMD_avx256_mrxnr_fp32( char, int, int, int, float, float *, float *, float, float *, int ); 
void gemm_microkernel_Cresident_AMD_avx256_8x8_fp32( char, int, int, int, float, float *, float *, float, float *, int );
void gemm_microkernel_Cresident_AMD_avx256_8x14_fp32(char, int, int, int, float, float *, float *, float, float *, int );
void gemm_microkernel_Cresident_AMD_avx256_16x6_optimum_fp32(char, int, int, int, float, float *, float *, float, float *, int );
void gemm_microkernel_Cresident_AMD_avx256_32x2_fp32( char, int, int, int, float, float *, float *, float, float *, int );
void gemm_microkernel_Cresident_AMD_avx256_40x2_fp32( char, int, int, int, float, float *, float *, float, float *, int );

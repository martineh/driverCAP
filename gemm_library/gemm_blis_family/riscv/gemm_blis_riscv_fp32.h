void gemm_microkernel_Cresident_riscv_4x4_prefetch_fp32( char, int, int, int, float, float *, float *, float, float *, int );
void gemm_microkernel_Cresident_riscv_4x4_prefetch_unroll_fp32( char, int, int, int, float, float *, float *, float, float *, int );
void gemm_microkernel_Cresident_riscv_8x8_prefetch_fp32( char, int, int, int, float, float *, float *, float, float *, int );
void gemm_microkernel_Cresident_riscv_8x8_prefetch_unroll_fp32( char, int, int, int, float, float *, float *, float, float *, int );

void gemm_microkernel_ABresident_riscv_4x4_fp32( char, char, int, int, int, float, float *, int, float *, float, float * );

//void fvtrans_float32_4x4( float32x4_t *, float32x4_t *, float32x4_t *, float32x4_t * );
void fvtrans_float32_4x4( vfloat32m1_t *, vfloat32m1_t *, vfloat32m1_t *, vfloat32m1_t * );
void fvtrans_float32_8x8( vfloat32m1_t *, vfloat32m1_t *, vfloat32m1_t *, vfloat32m1_t *,
                          vfloat32m1_t *, vfloat32m1_t *, vfloat32m1_t *, vfloat32m1_t *,
                          vfloat32m1_t *, vfloat32m1_t *, vfloat32m1_t *, vfloat32m1_t *,
                          vfloat32m1_t *, vfloat32m1_t *, vfloat32m1_t *, vfloat32m1_t * );


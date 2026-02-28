
#include <stdio.h>
#include <stdlib.h>
#include <arm_neon.h>

#define Acol(a1,a2)  A[ (a2)*(ldA)+(a1) ]
#define Bcol(a1,a2)  B[ (a2)*(ldB)+(a1) ]
#define Ccol(a1,a2)  C[ (a2)*(ldC)+(a1) ]
#define Ctcol(a1,a2) Ctmp[ (a2)*(ldCt)+(a1) ]

#define Arow(a1,a2)  A[ (a1)*(ldA)+(a2) ]
#define Brow(a1,a2)  B[ (a1)*(ldB)+(a2) ]
#define Crow(a1,a2)  C[ (a1)*(ldC)+(a2) ]
#define Ctrow(a1,a2) Ctmp[ (a1)*(ldCt)+(a2) ]

#define Ctref(a1,a2) Ctmp[ (a2)*(ldCt)+(a1) ]
#define Atref(a1,a2) Atmp[ (a2)*(Atlda)+(a1) ]

inline void gemm_microkernel_Cresident_neon_4x4_fp32( char orderC, int mr, int nr, int kc, float alpha, float *Ar, float *Br, float beta, float *C, int ldC ){

    int         i, j, k, baseA = 0, baseB = 0, ldCt = MR, Amr, Bnr;
    float32x4_t C00, C01, C02, C03, 
                A00, A01, A02, A03, 
                A10, A11, A12, A13, B0 ; 

{
#define A0    A00

    float  zero = 0.0, one = 1.0, *Aptr, *Bptr, Ctmp[MR*NR];
    
    if ( kc==0 ) return;

    C00 = vmovq_n_f32(0);
    C01 = vmovq_n_f32(0);
    C02 = vmovq_n_f32(0);
    C03 = vmovq_n_f32(0);
    
    Aptr = &Br[0];
    Bptr = &Ar[0];
    Amr  = NR;
    Bnr  = MR;
    
    if ( alpha!=zero ) {
	    for ( k=0; k<kc; k++ ) {
		    
		    A0 = vld1q_f32(&Aptr[baseA]);
		    B0 = vld1q_f32(&Bptr[baseB]);
		    
		    C00 = vfmaq_laneq_f32(C00, A0, B0, 0);
		    C01 = vfmaq_laneq_f32(C01, A0, B0, 1);
		    C02 = vfmaq_laneq_f32(C02, A0, B0, 2);
		    C03 = vfmaq_laneq_f32(C03, A0, B0, 3);
		    
		    baseA = baseA+Amr; 
		    baseB = baseB+Bnr;
	    }
	    
	    if ( alpha==-one ) {
		    C00 = -C00; C01 = -C01; C02 = -C02; C03 = -C03; 
	    }
	    else if ( alpha!=one ) {
		    C00 = alpha*C00; C01 = alpha*C01; C02 = alpha*C02; C03 = alpha*C03; 
	    }
    }
    
    if ( (mr<MR)||(nr<NR) ) {
	    vst1q_f32(&Ctref(0,0), C00);
	    vst1q_f32(&Ctref(0,1), C01);
	    vst1q_f32(&Ctref(0,2), C02);
	    vst1q_f32(&Ctref(0,3), C03);
	    
	    if ( beta!=zero ) {
	      for ( j=0; j<nr; j++ ) 
	        for ( i=0; i<mr; i++ ) 
	          Crow(i,j) = beta*Crow(i,j) + Ctrow(i,j);
		    
	    }
	    else {
	      for ( j=0; j<nr; j++ ) 
	        for ( i=0; i<mr; i++ ) 
	          Crow(i,j) = Ctrow(i,j);
	    }
    }
    else if ( (mr==MR)&&(nr==NR) ) {
	    if ( beta!=zero ) {
		    A00 = vld1q_f32(&Ccol(0,0));
		    A01 = vld1q_f32(&Ccol(0,1));
		    A02 = vld1q_f32(&Ccol(0,2));
		    A03 = vld1q_f32(&Ccol(0,3));
		    
		    C00 = beta*A00 + C00;
		    C01 = beta*A01 + C01;
		    C02 = beta*A02 + C02;
		    C03 = beta*A03 + C03;
		 
	    }
	    
	    vst1q_f32(&Ccol(0,0), C00);
	    vst1q_f32(&Ccol(0,1), C01);
	    vst1q_f32(&Ccol(0,2), C02);
	    vst1q_f32(&Ccol(0,3), C03);
	    
    }
    else {
	    printf("Error: Incorrect use of 4x4 micro-kernel with %d x %d block\n", mr, nr);
	    exit(-1);
    }
}
}




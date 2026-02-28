/* 
   GEMM FLAVOURS

   -----

   GEMM FLAVOURS is a family of algorithms for matrix multiplication based
   on the BLIS approach for this operation: https://github.com/flame/blis

   -----

   This program is free software: you can redistribute it and/or modify it under
   the terms of the GNU General Public License as published by the Free Software
   Foundation, either version 3 of the License, or (at your option) any later
   version.

   This program is distributed in the hope that it will be useful, but WITHOUT
   ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
   FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
   You should have received a copy of the GNU General Public License along with
   this program. If not, see <http://www.gnu.org/licenses/>.
   -----

   author    = "Enrique S. Quintana-Orti"
   contact   = "quintana@disca.upv.es"
   copyright = "Copyright 2021, Universitat Politecnica de Valencia"
   license   = "GPLv3"
   status    = "Production"
   version   = "1.1"
*/

#include <stdio.h>
#include <stdlib.h>
#include <arm_neon.h>

#include "blis.h"
#include "gemm_blis_neon_fp16.h"

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

inline void gemm_microkernel_Cresident_neon_8x8_prefetch_fp16( char orderC, int mr, int nr, int kc, _Float16 alpha, _Float16 *Ar, _Float16 *Br, _Float16 beta, _Float16 *C, int ldC ) {
/*
  BLIS GEMM microkernel, computes the product Cr := Cr + Ar * Br
//Specific: only for MRxNR = 4x4
*/
    printf("Error: not implemented %d %d\n", mr, nr);
    exit(-1);
/*
  int         i, j, k, baseA, baseB, ldCt = MR, Amr, Bnr;
  float16x8_t C0, C1, C2, C3, C4, C5, C6, C7, C8,
	      A0, A1, A2, A3, A4, A5, A6, A7, A0n, B0, B0n;
  _Float16    zero = 0.0, one = 1.0, *Aptr, *Bptr, Ctmp[MR*NR];

  if ( kc==0 )
    return;

  C0 = vmovq_n_f16(0);
  C1 = vmovq_n_f16(0);
  C2 = vmovq_n_f16(0);
  C3 = vmovq_n_f16(0);
  C4 = vmovq_n_f16(0);
  C5 = vmovq_n_f16(0);
  C6 = vmovq_n_f16(0);
  C7 = vmovq_n_f16(0);

  if ( orderC=='C' ) {
    Aptr = &Ar[0];
    Bptr = &Br[0];
    Amr  = MR;
    Bnr  = NR;
  }
  else {
    Aptr = &Br[0];
    Bptr = &Ar[0];
    Amr  = NR;
    Bnr  = MR;
  }
  A0 = vld1q_f16(&Aptr[0]);
  B0 = vld1q_f16(&Bptr[0]);

  // Iterate from 1 to kc-1 (last iteration outside loop)
  if ( alpha!=zero ) {
    for ( k=0; k<kc-1; k++ ) {

      // Prefect colum/row of A/B for next iteration
      baseA = baseA+Amr;
      baseB = baseB+Bnr;

      A0n = vld1q_f16(&Aptr[baseA]);
      B0n = vld1q_f16(&Bptr[baseB]);
       
      C0 = vfmaq_laneq_f16(C0, A0, B0, 0);
      C1 = vfmaq_laneq_f16(C1, A0, B0, 1);
      C2 = vfmaq_laneq_f16(C2, A0, B0, 2);
      C3 = vfmaq_laneq_f16(C3, A0, B0, 3);
      C4 = vfmaq_laneq_f16(C4, A0, B0, 4);
      C5 = vfmaq_laneq_f16(C5, A0, B0, 5);
      C6 = vfmaq_laneq_f16(C6, A0, B0, 6);
      C7 = vfmaq_laneq_f16(C7, A0, B0, 7);
    
      A0 = A0n;
      B0 = B0n;
    }
  
    // Last iteration
    C0 = vfmaq_laneq_f16(C0, A0, B0, 0);
    C1 = vfmaq_laneq_f16(C1, A0, B0, 1);
    C2 = vfmaq_laneq_f16(C2, A0, B0, 2);
    C3 = vfmaq_laneq_f16(C3, A0, B0, 3);
    C4 = vfmaq_laneq_f16(C4, A0, B0, 4);
    C5 = vfmaq_laneq_f16(C5, A0, B0, 5);
    C6 = vfmaq_laneq_f16(C6, A0, B0, 6);
    C7 = vfmaq_laneq_f16(C7, A0, B0, 7);
  
    if ( alpha==-one ) {
      C0 = -C0; C1 = -C1; C2 = -C2; C3 = -C3; 
      C4 = -C4; C5 = -C5; C6 = -C6; C7 = -C7; 
    }
    else if ( alpha!=one ) {
      C0 = alpha*C0; C1 = alpha*C1; C2 = alpha*C2; C3 = alpha*C3;
      C4 = alpha*C4; C5 = alpha*C5; C6 = alpha*C6; C7 = alpha*C7;
    }
  }

  if ( (mr<MR)||(nr<NR) ) {
    vst1q_f16(&Ctref(0,0), C0);
    vst1q_f16(&Ctref(0,1), C1);
    vst1q_f16(&Ctref(0,2), C2);
    vst1q_f16(&Ctref(0,3), C3);
    vst1q_f16(&Ctref(0,4), C4);
    vst1q_f16(&Ctref(0,5), C5);
    vst1q_f16(&Ctref(0,6), C6);
    vst1q_f16(&Ctref(0,7), C7);
    if ( beta!=zero ) {
      if ( orderC=='C' ) {
        for ( j=0; j<nr; j++ ) 
          for ( i=0; i<mr; i++ ) 
            Ccol(i,j) = beta*Ccol(i,j) + Ctcol(i,j);
      }
      else {
        for ( j=0; j<nr; j++ ) 
          for ( i=0; i<mr; i++ ) 
            Crow(i,j) = beta*Crow(i,j) + Ctrow(i,j);
      }
    }
    else {
      if ( orderC=='C' ) {
        for ( j=0; j<nr; j++ ) 
          for ( i=0; i<mr; i++ ) 
            Ccol(i,j) = Ctcol(i,j);
      }
      else {
        for ( j=0; j<nr; j++ ) 
          for ( i=0; i<mr; i++ ) 
            Crow(i,j) = Ctrow(i,j);
      }
    }
  }
  else if ( (mr==MR)&&(nr==NR) ) {
    if ( beta!=zero ) {
      A0 = vld1q_f16(&Ccol(0,0));
      A1 = vld1q_f16(&Ccol(0,1));
      A2 = vld1q_f16(&Ccol(0,2));
      A3 = vld1q_f16(&Ccol(0,3));
      A4 = vld1q_f16(&Ccol(0,4));
      A5 = vld1q_f16(&Ccol(0,5));
      A6 = vld1q_f16(&Ccol(0,6));
      A7 = vld1q_f16(&Ccol(0,7));
      
      C0 = beta*A0 + C0;
      C1 = beta*A1 + C1;
      C2 = beta*A2 + C2;
      C3 = beta*A3 + C3;
      C4 = beta*A4 + C4;
      C5 = beta*A5 + C5;
      C6 = beta*A6 + C6;
      C7 = beta*A7 + C7;
    }

    vst1q_f16(&Ccol(0,0), C0);
    vst1q_f16(&Ccol(0,1), C1);
    vst1q_f16(&Ccol(0,2), C2);
    vst1q_f16(&Ccol(0,3), C3);
    vst1q_f16(&Ccol(0,4), C4);
    vst1q_f16(&Ccol(0,5), C5);
    vst1q_f16(&Ccol(0,6), C6);
    vst1q_f16(&Ccol(0,7), C7);
  }
  else {
    printf("Error: Incorrect use of 4x4 micro-kernel with %d x %d block\n", mr, nr);
    exit(-1);
  }
*/
}

inline void gemm_microkernel_Cresident_neon_4x4_prefetch_fp16( char orderC, int mr, int nr, int kc, _Float16 alpha, _Float16 *Ar, _Float16 *Br, _Float16 beta, _Float16 *C, int ldC ) {
    printf("Error: not implemented %d %d\n", mr, nr);
    exit(-1);
}

inline void gemm_microkernel_ABresident_neon_4x4_prefetch_fp16( char orderA, char transA, int mr, int nr, int kc, _Float16 alpha, _Float16 *A, int ldA, _Float16 *Br, _Float16 beta, _Float16 *Cr ) {
    printf("Error: not implemented %d %d\n", mr, nr);
    exit(-1);
}

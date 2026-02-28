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
#include "gemm_blis_neon_fp64.h"

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

inline void gemm_microkernel_Cresident_neon_4x4_prefetch_fp64( char orderC, int mr, int nr, int kc, double alpha, double *Ar, double *Br, double beta, double *C, int ldC ) {
/*
  BLIS GEMM microkernel, computes the product Cr := Cr + Ar * Br
  Specific: only for MRxNR = 4x4
*/
  int         i, j, k, baseA, baseB, ldCt = MR, Amr, Bnr;
  float64x2_t C00, C01, C02, C03, 
              C10, C11, C12, C13, 
              A00, A01, A02, A03, 
              A10, A11, A12, A13, 
              A0, A0n, A1, A1n, B0, B0n, B1, B1n;
  double      zero = 0.0, one = 1.0, *Aptr, *Bptr, Ctmp[MR*NR];

  if ( kc==0 )
    return;

  C00 = vmovq_n_f64(0);
  C01 = vmovq_n_f64(0);
  C02 = vmovq_n_f64(0);
  C03 = vmovq_n_f64(0);

  C10 = vmovq_n_f64(0);
  C11 = vmovq_n_f64(0);
  C12 = vmovq_n_f64(0);
  C13 = vmovq_n_f64(0);

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
  A0 = vld1q_f64(&Aptr[0]);
  A1 = vld1q_f64(&Aptr[2]);
  B0 = vld1q_f64(&Bptr[0]);
  B1 = vld1q_f64(&Bptr[2]);

  // Iterate from 1 to kc-1 (last iteration outside loop)
  if ( alpha!=zero ) {
    for ( k=0; k<kc-1; k++ ) {

      // Prefect colum/row of A/B for next iteration
      baseA = baseA+Amr;
      baseB = baseB+Bnr;

      A0n = vld1q_f64(&Aptr[baseA]);
      A1n = vld1q_f64(&Aptr[baseA+2]);
      B0n = vld1q_f64(&Bptr[baseB]);
      B1n = vld1q_f64(&Bptr[baseB+2]);
       
      C00 = vfmaq_laneq_f64(C00, A0, B0, 0);
      C01 = vfmaq_laneq_f64(C01, A0, B0, 1);
      C02 = vfmaq_laneq_f64(C02, A0, B1, 0);
      C03 = vfmaq_laneq_f64(C03, A0, B1, 1);
    
      C10 = vfmaq_laneq_f64(C10, A1, B0, 0);
      C11 = vfmaq_laneq_f64(C11, A1, B0, 1);
      C12 = vfmaq_laneq_f64(C12, A1, B1, 0);
      C13 = vfmaq_laneq_f64(C13, A1, B1, 1);
    
      A0 = A0n;
      A1 = A1n;
      B0 = B0n;
      B1 = B1n;
    }
  
    // Last iteration
    C00 = vfmaq_laneq_f64(C00, A0, B0, 0);
    C01 = vfmaq_laneq_f64(C01, A0, B0, 1);
    C02 = vfmaq_laneq_f64(C02, A0, B1, 0);
    C03 = vfmaq_laneq_f64(C03, A0, B1, 1);
    
    C10 = vfmaq_laneq_f64(C10, A1, B0, 0);
    C11 = vfmaq_laneq_f64(C11, A1, B0, 1);
    C12 = vfmaq_laneq_f64(C12, A1, B1, 0);
    C13 = vfmaq_laneq_f64(C13, A1, B1, 1);
  
    if ( alpha==-one ) {
      C00 = -C00; C01 = -C01; C02 = -C02; C03 = -C03; 
      C10 = -C10; C11 = -C11; C12 = -C12; C13 = -C13; 
    }
    else if ( alpha!=one ) {
      C00 = alpha*C00; C01 = alpha*C01; C02 = alpha*C02; C03 = alpha*C03;
      C10 = alpha*C10; C11 = alpha*C11; C12 = alpha*C12; C13 = alpha*C13;
    }
  }

  if ( (mr<MR)||(nr<NR) ) {
    vst1q_f64(&Ctref(0,0), C00);
    vst1q_f64(&Ctref(0,1), C01);
    vst1q_f64(&Ctref(0,2), C02);
    vst1q_f64(&Ctref(0,3), C03);
    vst1q_f64(&Ctref(2,0), C10);
    vst1q_f64(&Ctref(2,1), C11);
    vst1q_f64(&Ctref(2,2), C12);
    vst1q_f64(&Ctref(2,3), C13);
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
      A00 = vld1q_f64(&Ccol(0,0));
      A01 = vld1q_f64(&Ccol(0,1));
      A02 = vld1q_f64(&Ccol(0,2));
      A03 = vld1q_f64(&Ccol(0,3));

      A10 = vld1q_f64(&Ccol(2,0));
      A11 = vld1q_f64(&Ccol(2,1));
      A12 = vld1q_f64(&Ccol(2,2));
      A13 = vld1q_f64(&Ccol(2,3));
      
      C00 = beta*A00 + C00;
      C01 = beta*A01 + C01;
      C02 = beta*A02 + C02;
      C03 = beta*A03 + C03;
      
      C10 = beta*A10 + C10;
      C11 = beta*A11 + C11;
      C12 = beta*A12 + C12;
      C13 = beta*A13 + C13;
    }

    vst1q_f64(&Ccol(0,0), C00);
    vst1q_f64(&Ccol(0,1), C01);
    vst1q_f64(&Ccol(0,2), C02);
    vst1q_f64(&Ccol(0,3), C03);

    vst1q_f64(&Ccol(2,0), C10);
    vst1q_f64(&Ccol(2,1), C11);
    vst1q_f64(&Ccol(2,2), C12);
    vst1q_f64(&Ccol(2,3), C13);
  }
  else {
    printf("Error: Incorrect use of 4x4 micro-kernel with %d x %d block\n", mr, nr);
    exit(-1);
  }
}

inline void gemm_microkernel_Cresident_neon_8x8_prefetch_fp64( char orderC, int mr, int nr, int kc, double alpha, double *Ar, double *Br, double beta, double *C, int ldC ) {
    printf("Error: not implemented %d %d\n", mr, nr);
    exit(-1);
}

inline void gemm_microkernel_ABresident_neon_4x4_prefetch_fp64( char orderA, char transA, int mr, int nr, int kc, double alpha, double *A, int ldA, double *Br, double beta, double *Cr ) {
    printf("Error: not implemented %d %d\n", mr, nr);
    exit(-1);
}

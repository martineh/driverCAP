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

#include "gemm_blis_amd_avx256_fp32.h"

#define max(a,b)     ( (a) < (b) ? (b) : (a) )
#define dabs(a)      ( (a) > 0.0 ? (a) : -(a) )
#define min(a,b)     ( (a) > (b) ? (b) : (a) )

#define Acol(a1,a2)  A[ (a2)*(ldA)+(a1) ]
#define Bcol(a1,a2)  B[ (a2)*(ldB)+(a1) ]
#define Ccol(a1,a2)  C[ (a2)*(ldC)+(a1) ]
#define Ctcol(a1,a2) Ctmp[ (a2)*(ldCt)+(a1) ]

#define Arow(a1,a2)  A[ (a1)*(ldA)+(a2) ]
#define Brow(a1,a2)  B[ (a1)*(ldB)+(a2) ]
#define Crow(a1,a2)  C[ (a1)*(ldC)+(a2) ]

#define Arrow(a1,a2) Ar[ (a1)*(ldA)+(a2) ]
#define Ctrow(a1,a2) Ctmp[ (a1)*(ldCt)+(a2) ]
#define Atrow(a1,a2) Atmp[ (a1)*(ldAt)+(a2) ]

#define Ctref(a1,a2) Ctmp[ (a2)*(ldCt)+(a1) ]
#define Atref(a1,a2) Atmp[ (a2)*(Atlda)+(a1) ]

#define LOOP_BODY { \
        B0 = _mm256_loadu_ps(&Bptr[baseB + 0]); \
        B1 = _mm256_loadu_ps(&Bptr[baseB + 8]); \
 \
        A0 = _mm256_broadcast_ss(&Aptr[baseA + 0 * ldA]); \
        C00 = _mm256_fmadd_ps(B0, A0, C00); \
        C10 = _mm256_fmadd_ps(B1, A0, C10); \
 \
        A0 = _mm256_broadcast_ss(&Aptr[baseA + 1 * ldA]); \
        C01 = _mm256_fmadd_ps(B0, A0, C01); \
        C11 = _mm256_fmadd_ps(B1, A0, C11); \
 \
        A0 = _mm256_broadcast_ss(&Aptr[baseA + 2 * ldA]); \
        C02 = _mm256_fmadd_ps(B0, A0, C02); \
        C12 = _mm256_fmadd_ps(B1, A0, C12); \
 \
        A0 = _mm256_broadcast_ss(&Aptr[baseA + 3 * ldA]); \
        C03 = _mm256_fmadd_ps(B0, A0, C03); \
        C13 = _mm256_fmadd_ps(B1, A0, C13); \
 \
        A0 = _mm256_broadcast_ss(&Aptr[baseA + 4 * ldA]); \
        C04 = _mm256_fmadd_ps(B0, A0, C04); \
        C14 = _mm256_fmadd_ps(B1, A0, C14); \
 \
        A0 = _mm256_broadcast_ss(&Aptr[baseA + 5 * ldA]); \
        C05 = _mm256_fmadd_ps(B0, A0, C05); \
        C15 = _mm256_fmadd_ps(B1, A0, C15); \
 \
        baseA = baseA + 1; \
        baseB = baseB + Bnr; \
\
}

void gemm_microkernel_Cresident_AMD_avx256_6x2v_nopack_unroll_fp32( char orderC, int mr, int nr, int kc, float alpha, 
		                                                   float *A, int ldA, float *Br, float beta, float *C, int ldC ) {

  int i, j, k, baseB = 0, baseA = 0, ldCt = NR, Amr, Bnr;
  __m256 C00, C01, C02, C03, C04, C05;
  __m256 C10, C11, C12, C13, C14, C15;
  
  __m256 B0, B1, A0;

  float   zero = 0.0, *Aptr, *Bptr, Ctmp[MR*NR];

  if ( kc==0 )
    return;

  if ( beta!=zero ) {
    
    C00 = _mm256_loadu_ps(&Crow(0, 0));
    C01 = _mm256_loadu_ps(&Crow(1, 0));
    C02 = _mm256_loadu_ps(&Crow(2, 0));
    C03 = _mm256_loadu_ps(&Crow(3, 0));
    C04 = _mm256_loadu_ps(&Crow(4, 0));
    C05 = _mm256_loadu_ps(&Crow(5, 0));

    C10 = _mm256_loadu_ps(&Crow(0, 8));
    C11 = _mm256_loadu_ps(&Crow(1, 8));
    C12 = _mm256_loadu_ps(&Crow(2, 8));
    C13 = _mm256_loadu_ps(&Crow(3, 8));
    C14 = _mm256_loadu_ps(&Crow(4, 8));
    C15 = _mm256_loadu_ps(&Crow(5, 8));

  } else {
    C00  = _mm256_set1_ps ( 0 ); 
    C01  = _mm256_set1_ps ( 0 ); 
    C02  = _mm256_set1_ps ( 0 ); 
    C03  = _mm256_set1_ps ( 0 ); 
    C04  = _mm256_set1_ps ( 0 ); 
    C05  = _mm256_set1_ps ( 0 ); 
  
    C10  = _mm256_set1_ps ( 0 ); 
    C11  = _mm256_set1_ps ( 0 ); 
    C12  = _mm256_set1_ps ( 0 ); 
    C13  = _mm256_set1_ps ( 0 ); 
    C14  = _mm256_set1_ps ( 0 ); 
    C15  = _mm256_set1_ps ( 0 ); 
  }

  Aptr = &A[0];
  Bptr = &Br[0];
  Bnr = NR;
  //Bnr  = NR * 4;

  if ( alpha!=zero ) {
    for (k = 0; k < kc % 4; k++)
      LOOP_BODY;
    for (k = kc % 4; k < kc; k += 4) {
      LOOP_BODY;
      LOOP_BODY;
      LOOP_BODY;
      LOOP_BODY;
    }
/*
    for (k = 0; k < kc; k += 4) {
      // Iter 0
      B0 = _mm256_loadu_ps(&Bptr[baseB + 0]);
      B1 = _mm256_loadu_ps(&Bptr[baseB + 8]);
        
      A0 = _mm256_broadcast_ss(&Aptr[baseA + 0 * ldA]);
      C00 = _mm256_fmadd_ps(B0, A0, C00);
      C10 = _mm256_fmadd_ps(B1, A0, C10);
        
      A0 = _mm256_broadcast_ss(&Aptr[baseA + 1 * ldA]);
      C01 = _mm256_fmadd_ps(B0, A0, C01);
      C11 = _mm256_fmadd_ps(B1, A0, C11);
      
      A0 = _mm256_broadcast_ss(&Aptr[baseA + 2 * ldA]);
      C02 = _mm256_fmadd_ps(B0, A0, C02);
      C12 = _mm256_fmadd_ps(B1, A0, C12);

      A0 = _mm256_broadcast_ss(&Aptr[baseA + 3 * ldA]);
      C03 = _mm256_fmadd_ps(B0, A0, C03);
      C13 = _mm256_fmadd_ps(B1, A0, C13);
      
      A0 = _mm256_broadcast_ss(&Aptr[baseA + 4 * ldA]);
      C04 = _mm256_fmadd_ps(B0, A0, C04);
      C14 = _mm256_fmadd_ps(B1, A0, C14);
      
      A0 = _mm256_broadcast_ss(&Aptr[baseA + 5 * ldA]);
      C05 = _mm256_fmadd_ps(B0, A0, C05);
      C15 = _mm256_fmadd_ps(B1, A0, C15);
        
      // Iter 1
      B0 = _mm256_loadu_ps(&Bptr[baseB + 16]);
      B1 = _mm256_loadu_ps(&Bptr[baseB + 24]);
        
      A0 = _mm256_broadcast_ss(&Aptr[baseA + 1 * ldA]);
      C00 = _mm256_fmadd_ps(B0, A0, C00);
      C10 = _mm256_fmadd_ps(B1, A0, C10);
        
      A0 = _mm256_broadcast_ss(&Aptr[baseA + 2 * ldA]);
      C01 = _mm256_fmadd_ps(B0, A0, C01);
      C11 = _mm256_fmadd_ps(B1, A0, C11);
      
      A0 = _mm256_broadcast_ss(&Aptr[baseA + 3 * ldA]);
      C02 = _mm256_fmadd_ps(B0, A0, C02);
      C12 = _mm256_fmadd_ps(B1, A0, C12);

      A0 = _mm256_broadcast_ss(&Aptr[baseA + 4 * ldA]);
      C03 = _mm256_fmadd_ps(B0, A0, C03);
      C13 = _mm256_fmadd_ps(B1, A0, C13);
      
      A0 = _mm256_broadcast_ss(&Aptr[baseA + 5 * ldA]);
      C04 = _mm256_fmadd_ps(B0, A0, C04);
      C14 = _mm256_fmadd_ps(B1, A0, C14);
      
      A0 = _mm256_broadcast_ss(&Aptr[baseA + 6 * ldA]);
      C05 = _mm256_fmadd_ps(B0, A0, C05);

      // Iter 2
      B0 = _mm256_loadu_ps(&Bptr[baseB + 32]);
      B1 = _mm256_loadu_ps(&Bptr[baseB + 40]);
        
      A0 = _mm256_broadcast_ss(&Aptr[baseA + 2 * ldA]);
      C00 = _mm256_fmadd_ps(B0, A0, C00);
      C10 = _mm256_fmadd_ps(B1, A0, C10);
        
      A0 = _mm256_broadcast_ss(&Aptr[baseA + 3 * ldA]);
      C01 = _mm256_fmadd_ps(B0, A0, C01);
      C11 = _mm256_fmadd_ps(B1, A0, C11);
      
      A0 = _mm256_broadcast_ss(&Aptr[baseA + 4 * ldA]);
      C02 = _mm256_fmadd_ps(B0, A0, C02);
      C12 = _mm256_fmadd_ps(B1, A0, C12);

      A0 = _mm256_broadcast_ss(&Aptr[baseA + 5 * ldA]);
      C03 = _mm256_fmadd_ps(B0, A0, C03);
      C13 = _mm256_fmadd_ps(B1, A0, C13);
      
      A0 = _mm256_broadcast_ss(&Aptr[baseA + 6 * ldA]);
      C04 = _mm256_fmadd_ps(B0, A0, C04);
      C14 = _mm256_fmadd_ps(B1, A0, C14);
      
      A0 = _mm256_broadcast_ss(&Aptr[baseA + 7 * ldA]);
      C05 = _mm256_fmadd_ps(B0, A0, C05);
      
      // Iter 3
      B0 = _mm256_loadu_ps(&Bptr[baseB + 48]);
      B1 = _mm256_loadu_ps(&Bptr[baseB + 56]);
        
      A0 = _mm256_broadcast_ss(&Aptr[baseA + 3 * ldA]);
      C00 = _mm256_fmadd_ps(B0, A0, C00);
      C10 = _mm256_fmadd_ps(B1, A0, C10);
        
      A0 = _mm256_broadcast_ss(&Aptr[baseA + 4 * ldA]);
      C01 = _mm256_fmadd_ps(B0, A0, C01);
      C11 = _mm256_fmadd_ps(B1, A0, C11);
      
      A0 = _mm256_broadcast_ss(&Aptr[baseA + 5 * ldA]);
      C02 = _mm256_fmadd_ps(B0, A0, C02);
      C12 = _mm256_fmadd_ps(B1, A0, C12);

      A0 = _mm256_broadcast_ss(&Aptr[baseA + 6 * ldA]);
      C03 = _mm256_fmadd_ps(B0, A0, C03);
      C13 = _mm256_fmadd_ps(B1, A0, C13);
      
      A0 = _mm256_broadcast_ss(&Aptr[baseA + 7 * ldA]);
      C04 = _mm256_fmadd_ps(B0, A0, C04);
      C14 = _mm256_fmadd_ps(B1, A0, C14);
      
      A0 = _mm256_broadcast_ss(&Aptr[baseA + 8 * ldA]);
      C05 = _mm256_fmadd_ps(B0, A0, C05);
      
      baseA = baseA + 4;	
      baseB = baseB + Bnr;
    }

    for (k = 0; k < kc % 4; k++) {
      B0 = _mm256_loadu_ps(&Bptr[baseB + 0]);
      B1 = _mm256_loadu_ps(&Bptr[baseB + 8]);
        
      A0 = _mm256_broadcast_ss(&Aptr[baseA + 0 * ldA]);
      C00 = _mm256_fmadd_ps(B0, A0, C00);
      C10 = _mm256_fmadd_ps(B1, A0, C10);
        
      A0 = _mm256_broadcast_ss(&Aptr[baseA + 1 * ldA]);
      C01 = _mm256_fmadd_ps(B0, A0, C01);
      C11 = _mm256_fmadd_ps(B1, A0, C11);
      
      A0 = _mm256_broadcast_ss(&Aptr[baseA + 2 * ldA]);
      C02 = _mm256_fmadd_ps(B0, A0, C02);
      C12 = _mm256_fmadd_ps(B1, A0, C12);

      A0 = _mm256_broadcast_ss(&Aptr[baseA + 3 * ldA]);
      C03 = _mm256_fmadd_ps(B0, A0, C03);
      C13 = _mm256_fmadd_ps(B1, A0, C13);
      
      A0 = _mm256_broadcast_ss(&Aptr[baseA + 4 * ldA]);
      C04 = _mm256_fmadd_ps(B0, A0, C04);
      C14 = _mm256_fmadd_ps(B1, A0, C14);
      
      A0 = _mm256_broadcast_ss(&Aptr[baseA + 5 * ldA]);
      C05 = _mm256_fmadd_ps(B0, A0, C05);
      C15 = _mm256_fmadd_ps(B1, A0, C15);
      
      baseA = baseA + 1;	
      baseB = baseB + NR;
    }
    */
  }
  
  if ((mr < MR) || (nr < NR)) {
   
    _mm256_storeu_ps(&Ctrow(0, 0), C00); 
    _mm256_storeu_ps(&Ctrow(1, 0), C01); 
    _mm256_storeu_ps(&Ctrow(2, 0), C02);
    _mm256_storeu_ps(&Ctrow(3, 0), C03);
    _mm256_storeu_ps(&Ctrow(4, 0), C04);
    _mm256_storeu_ps(&Ctrow(5, 0), C05);

    _mm256_storeu_ps(&Ctrow(0, 8), C10); 
    _mm256_storeu_ps(&Ctrow(1, 8), C11); 
    _mm256_storeu_ps(&Ctrow(2, 8), C12);
    _mm256_storeu_ps(&Ctrow(3, 8), C13);
    _mm256_storeu_ps(&Ctrow(4, 8), C14);
    _mm256_storeu_ps(&Ctrow(5, 8), C15);

     for ( j=0; j<nr; j++ )
       for ( i=0; i<mr; i++ )
         Crow(i,j) = Ctrow(i,j);
  
  } else {

    _mm256_storeu_ps(&Crow(0, 0), C00);
    _mm256_storeu_ps(&Crow(1, 0), C01);
    _mm256_storeu_ps(&Crow(2, 0), C02);
    _mm256_storeu_ps(&Crow(3, 0), C03);
    _mm256_storeu_ps(&Crow(4, 0), C04);
    _mm256_storeu_ps(&Crow(5, 0), C05);
    
    _mm256_storeu_ps(&Crow(0, 8), C10);
    _mm256_storeu_ps(&Crow(1, 8), C11);
    _mm256_storeu_ps(&Crow(2, 8), C12);
    _mm256_storeu_ps(&Crow(3, 8), C13);
    _mm256_storeu_ps(&Crow(4, 8), C14);
    _mm256_storeu_ps(&Crow(5, 8), C15);
     
  }

}

void gemm_microkernel_Cresident_AMD_avx256_6x2v_nopack_fp32( char orderC, int mr, int nr, int kc, float alpha, 
		                                             float *A, int ldA, float *Br, float beta, float *C, int ldC ) {

  int i, j, k, baseB = 0, baseA = 0, ldCt = NR, Amr, Bnr;
  __m256 C00, C01, C02, C03, C04, C05;
  __m256 C10, C11, C12, C13, C14, C15;
  
  __m256 B0, B1, A0;

  float   zero = 0.0, *Aptr, *Bptr, Ctmp[MR*NR];

  if ( kc==0 )
    return;

  if ( beta!=zero ) {
    
    C00 = _mm256_loadu_ps(&Crow(0, 0));
    C01 = _mm256_loadu_ps(&Crow(1, 0));
    C02 = _mm256_loadu_ps(&Crow(2, 0));
    C03 = _mm256_loadu_ps(&Crow(3, 0));
    C04 = _mm256_loadu_ps(&Crow(4, 0));
    C05 = _mm256_loadu_ps(&Crow(5, 0));

    C10 = _mm256_loadu_ps(&Crow(0, 8));
    C11 = _mm256_loadu_ps(&Crow(1, 8));
    C12 = _mm256_loadu_ps(&Crow(2, 8));
    C13 = _mm256_loadu_ps(&Crow(3, 8));
    C14 = _mm256_loadu_ps(&Crow(4, 8));
    C15 = _mm256_loadu_ps(&Crow(5, 8));

  } else {
    C00  = _mm256_set1_ps ( 0 ); 
    C01  = _mm256_set1_ps ( 0 ); 
    C02  = _mm256_set1_ps ( 0 ); 
    C03  = _mm256_set1_ps ( 0 ); 
    C04  = _mm256_set1_ps ( 0 ); 
    C05  = _mm256_set1_ps ( 0 ); 
  
    C10  = _mm256_set1_ps ( 0 ); 
    C11  = _mm256_set1_ps ( 0 ); 
    C12  = _mm256_set1_ps ( 0 ); 
    C13  = _mm256_set1_ps ( 0 ); 
    C14  = _mm256_set1_ps ( 0 ); 
    C15  = _mm256_set1_ps ( 0 ); 
  }

  Aptr = &A[0];
  Bptr = &Br[0];
  Bnr  = NR;

  if ( alpha!=zero ) {
    //if (mr == MR && nr == NR) 
      for ( k=0; k < kc; k++ ) {
        // Load colums/rows of A/B for current iteration
        B0 = _mm256_loadu_ps(&Bptr[baseB + 0]);
        B1 = _mm256_loadu_ps(&Bptr[baseB + 8]);
        
        A0 = _mm256_broadcast_ss(&Aptr[baseA + 0 * ldA]);
        C00 = _mm256_fmadd_ps(B0, A0, C00);
        C10 = _mm256_fmadd_ps(B1, A0, C10);
        
        A0 = _mm256_broadcast_ss(&Aptr[baseA + 1 * ldA]);
        C01 = _mm256_fmadd_ps(B0, A0, C01);
        C11 = _mm256_fmadd_ps(B1, A0, C11);
      
        A0 = _mm256_broadcast_ss(&Aptr[baseA + 2 * ldA]);
        C02 = _mm256_fmadd_ps(B0, A0, C02);
        C12 = _mm256_fmadd_ps(B1, A0, C12);

        A0 = _mm256_broadcast_ss(&Aptr[baseA + 3 * ldA]);
        C03 = _mm256_fmadd_ps(B0, A0, C03);
        C13 = _mm256_fmadd_ps(B1, A0, C13);
      
        A0 = _mm256_broadcast_ss(&Aptr[baseA + 4 * ldA]);
        C04 = _mm256_fmadd_ps(B0, A0, C04);
        C14 = _mm256_fmadd_ps(B1, A0, C14);
      
        A0 = _mm256_broadcast_ss(&Aptr[baseA + 5 * ldA]);
        C05 = _mm256_fmadd_ps(B0, A0, C05);
        C15 = _mm256_fmadd_ps(B1, A0, C15);
        
        baseA = baseA + 1;	
        baseB = baseB + Bnr;
      }
  }

  if ((mr < MR) || (nr < NR)) {
   
    _mm256_storeu_ps(&Ctrow(0, 0), C00); 
    _mm256_storeu_ps(&Ctrow(1, 0), C01); 
    _mm256_storeu_ps(&Ctrow(2, 0), C02);
    _mm256_storeu_ps(&Ctrow(3, 0), C03);
    _mm256_storeu_ps(&Ctrow(4, 0), C04);
    _mm256_storeu_ps(&Ctrow(5, 0), C05);

    _mm256_storeu_ps(&Ctrow(0, 8), C10); 
    _mm256_storeu_ps(&Ctrow(1, 8), C11); 
    _mm256_storeu_ps(&Ctrow(2, 8), C12);
    _mm256_storeu_ps(&Ctrow(3, 8), C13);
    _mm256_storeu_ps(&Ctrow(4, 8), C14);
    _mm256_storeu_ps(&Ctrow(5, 8), C15);

     for ( j=0; j<nr; j++ )
       for ( i=0; i<mr; i++ )
         Crow(i,j) = Ctrow(i,j);
  
  } else {

    _mm256_storeu_ps(&Crow(0, 0), C00);
    _mm256_storeu_ps(&Crow(1, 0), C01);
    _mm256_storeu_ps(&Crow(2, 0), C02);
    _mm256_storeu_ps(&Crow(3, 0), C03);
    _mm256_storeu_ps(&Crow(4, 0), C04);
    _mm256_storeu_ps(&Crow(5, 0), C05);
    
    _mm256_storeu_ps(&Crow(0, 8), C10);
    _mm256_storeu_ps(&Crow(1, 8), C11);
    _mm256_storeu_ps(&Crow(2, 8), C12);
    _mm256_storeu_ps(&Crow(3, 8), C13);
    _mm256_storeu_ps(&Crow(4, 8), C14);
    _mm256_storeu_ps(&Crow(5, 8), C15);
     
  }
}

void gemm_microkernel_Cresident_AMD_avx256_4x3v_nopack_fp32( char orderC, int mr, int nr, int kc, float alpha, 
		                                             float *A, int ldA, float *Br, float beta, float *C, int ldC ) {

  int i, j, k, baseB = 0, baseA = 0, ldCt = NR, Amr, Bnr;
  __m256 C00, C01, C02, C03,
         C10, C11, C12, C13,
         C20, C21, C22, C23;
  
  __m256 B0, B1, B2, A0;

  float   zero = 0.0, *Aptr, *Bptr, Ctmp[MR*NR];

  if ( kc==0 )
    return;

  if ( beta!=zero ) {
    
    C00 = _mm256_loadu_ps(&Crow(0, 0));
    C01 = _mm256_loadu_ps(&Crow(1, 0));
    C02 = _mm256_loadu_ps(&Crow(2, 0));
    C03 = _mm256_loadu_ps(&Crow(3, 0));

    C10 = _mm256_loadu_ps(&Crow(0, 8));
    C11 = _mm256_loadu_ps(&Crow(1, 8));
    C12 = _mm256_loadu_ps(&Crow(2, 8));
    C13 = _mm256_loadu_ps(&Crow(3, 8));

    C20 = _mm256_loadu_ps(&Crow(0, 16));
    C21 = _mm256_loadu_ps(&Crow(1, 16));
    C22 = _mm256_loadu_ps(&Crow(2, 16));
    C23 = _mm256_loadu_ps(&Crow(3, 16));
  } else {
    C00  = _mm256_set1_ps ( 0 ); 
    C01  = _mm256_set1_ps ( 0 ); 
    C02  = _mm256_set1_ps ( 0 ); 
    C03  = _mm256_set1_ps ( 0 ); 
  
    C10  = _mm256_set1_ps ( 0 ); 
    C11  = _mm256_set1_ps ( 0 ); 
    C12  = _mm256_set1_ps ( 0 ); 
    C13  = _mm256_set1_ps ( 0 ); 
    
    C20  = _mm256_set1_ps ( 0 ); 
    C21  = _mm256_set1_ps ( 0 ); 
    C22  = _mm256_set1_ps ( 0 ); 
    C23  = _mm256_set1_ps ( 0 ); 
  }

  Aptr = &A[0];
  Bptr = &Br[0];
  Bnr  = NR;

  if ( alpha!=zero ) {
    //if (mr == MR && nr == NR) 
      for ( k=0; k < kc; k++ ) {
        // Load colums/rows of A/B for current iteration
        B0 = _mm256_loadu_ps(&Bptr[baseB + 0]);
        B1 = _mm256_loadu_ps(&Bptr[baseB + 8]);
        B2 = _mm256_loadu_ps(&Bptr[baseB + 16]);
        
        A0 = _mm256_broadcast_ss(&Aptr[baseA + 0 * ldA]);
        C00 = _mm256_fmadd_ps(B0, A0, C00);
        C10 = _mm256_fmadd_ps(B1, A0, C10);
        C20 = _mm256_fmadd_ps(B2, A0, C20);
        
        A0 = _mm256_broadcast_ss(&Aptr[baseA + 1 * ldA]);
        C01 = _mm256_fmadd_ps(B0, A0, C01);
        C11 = _mm256_fmadd_ps(B1, A0, C11);
        C21 = _mm256_fmadd_ps(B2, A0, C21);
      
        A0 = _mm256_broadcast_ss(&Aptr[baseA + 2 * ldA]);
        C02 = _mm256_fmadd_ps(B0, A0, C02);
        C12 = _mm256_fmadd_ps(B1, A0, C12);
        C22 = _mm256_fmadd_ps(B2, A0, C22);

        A0 = _mm256_broadcast_ss(&Aptr[baseA + 3 * ldA]);
        C03 = _mm256_fmadd_ps(B0, A0, C03);
        C13 = _mm256_fmadd_ps(B1, A0, C13);
        C23 = _mm256_fmadd_ps(B2, A0, C23);
      
        baseA = baseA + 1;	
        baseB = baseB + Bnr;
      }
  }

  if ((mr < MR) || (nr < NR)) {
   
    _mm256_storeu_ps(&Ctrow(0, 0), C00); 
    _mm256_storeu_ps(&Ctrow(1, 0), C01); 
    _mm256_storeu_ps(&Ctrow(2, 0), C02);
    _mm256_storeu_ps(&Ctrow(3, 0), C03);

    _mm256_storeu_ps(&Ctrow(0, 8), C10); 
    _mm256_storeu_ps(&Ctrow(1, 8), C11); 
    _mm256_storeu_ps(&Ctrow(2, 8), C12);
    _mm256_storeu_ps(&Ctrow(3, 8), C13);

    _mm256_storeu_ps(&Ctrow(0, 16), C20); 
    _mm256_storeu_ps(&Ctrow(1, 16), C21); 
    _mm256_storeu_ps(&Ctrow(2, 16), C22);
    _mm256_storeu_ps(&Ctrow(3, 16), C23);
    
    for ( j=0; j<nr; j++ )
      for ( i=0; i<mr; i++ )
        Crow(i,j) = Ctrow(i,j);
  
  } else {

    _mm256_storeu_ps(&Crow(0, 0), C00);
    _mm256_storeu_ps(&Crow(1, 0), C01);
    _mm256_storeu_ps(&Crow(2, 0), C02);
    _mm256_storeu_ps(&Crow(3, 0), C03);
    
    _mm256_storeu_ps(&Crow(0, 8), C10);
    _mm256_storeu_ps(&Crow(1, 8), C11);
    _mm256_storeu_ps(&Crow(2, 8), C12);
    _mm256_storeu_ps(&Crow(3, 8), C13);
     
    _mm256_storeu_ps(&Crow(0, 16), C20);
    _mm256_storeu_ps(&Crow(1, 16), C21);
    _mm256_storeu_ps(&Crow(2, 16), C22);
    _mm256_storeu_ps(&Crow(3, 16), C23);
  }
}

/*
#define vl_fp32 8        // Tamaño del vector en fp32, o sea, 8 elementos de tipo fp32, que es lo que cabe en un __m512.
#define mrdiv (MR/vl_fp32)  
#define nrdiv (NR/vl_fp32)  
#define vregister __m256

#define vinit(vreg, n)    { vreg = _mm256_set1_ps(n); }
#define vload(vreg, mem)  { vreg = _mm256_loadu_ps(mem); }
#define vstore(mem, vreg) { _mm256_storeu_ps(mem, vreg); }

#define vupdate(vreg1, vreg2, vreg3, i) { vreg1 = _mm256_fmadd_ps ( vreg2, _mm256_set1_ps ( vreg3[i] ), vreg1 ); }
#define vadd(vreg1, vreg2, vreg3)       { vreg1 = _mm256_add_ps   ( vreg2, vreg3 ); }
//#define vupdate(vreg1, vreg2, vreg3, i) { vreg1 += vreg2*vreg3[i]; }


void gemm_microkernel_Cresident_AMD_avx256_mrxnr_fp32( char orderC, int mr, int nr, int kc, float alpha, 
		                                       float *Ar, float *Br, float beta, float *C, int ldC ) { 
// Micro-kernel with C resident in regs. Inputs:
// C  stored in column-major order, with leading dimension Clda
// Ar packed by columns, with leading dimension mr
// Br packed by rows, with leading dimension nr
// Ojo! Este kernel pasa del orderC

  int mrv = mrdiv;
  int nrv = nrdiv;

  if (MR % vl_fp32 != 0) mrv++;
  if (NR % vl_fp32 != 0) nrv++;
  
  int       iv, j, jv, pr, kv, baseA = 0, baseB = 0;
  vregister Cr[mrv][NR],      // Micro-tile of C
	    ar[mrv], br[nrv]; // Single column/row of Ar/Br

  float zero = 0.0;

  // Load micro-tile into vector registers
  if ((beta != zero) && 
     ((mr == MR) || (nr == NR))) {
    for ( j = 0; j < NR; j++ )
      for ( iv = 0; iv < mrv; iv++ )
        vload(Cr[iv][j], &C[j*ldC+iv*vl_fp32]);
  } else {
    for ( j = 0; j < nr; j++ )
      for ( iv = 0; iv < mrv; iv++ )
        vinit(Cr[iv][j], 0.0);
  }
  
  for (pr = 0; pr < kc; pr++ ) { 
    // Load pr-th column/row of Ar/Br into vector registers
    for (iv = 0; iv < mrv; iv++ )
      vload(ar[iv], &Ar[baseA + iv * vl_fp32]);
    for (jv = 0; jv < nrv; jv++ )
      vload(br[jv], &Br[baseB + jv * vl_fp32]);

    // Update the micro-tile with axpy
    for ( iv = 0; iv < mrv; iv++ )
      for ( jv = 0; jv < nrv; jv++ ) {
	int lim = min(vl_fp32, NR - jv * vl_fp32);
        for ( kv = 0; kv < lim; kv++ ) {
          vupdate(Cr[iv][jv * vl_fp32 + kv], ar[iv], br[jv], kv);
        }
      }

    baseA += MR; baseB += NR;
  }
  //(mr == MR && nr == NR)
    for ( j = 0; j < nr; j++ )
      for ( iv = 0; iv < mrv; iv++ )
        vstore(&C[j * ldC + iv * vl_fp32], Cr[iv][j]);
  //}

}
*/

void gemm_microkernel_Cresident_AMD_avx256_8x8_fp32( char orderC, int mr, int nr, int kc, float alpha, 
		                                     float *Ar, float *Br, float beta, float *C, int ldC ) {

  int i, j, k, baseB = 0, baseA = 0, ldCt = MR, Amr, Bnr;
  
  __m256 C00,  C01,   C02,  C03,  C04,  C05,  C06,  C07;
  __m256 A0;
  __m256 B0, B1;

  float   zero = 0.0, *Aptr, *Bptr, Ctmp[MR*NR];

  if ( kc==0 )
    return;

  C00 = _mm256_set1_ps ( 0 ); 
  C01 = _mm256_set1_ps ( 0 ); 
  C02 = _mm256_set1_ps ( 0 ); 
  C03 = _mm256_set1_ps ( 0 ); 
  C04 = _mm256_set1_ps ( 0 ); 
  C05 = _mm256_set1_ps ( 0 ); 
  C06 = _mm256_set1_ps ( 0 ); 
  C07 = _mm256_set1_ps ( 0 ); 

  if ( orderC=='C' ) {
    Aptr = &Ar[0];
    Bptr = &Br[0];
    Amr  = MR;
    Bnr  = NR;
  } else {
    Aptr = &Br[0];
    Bptr = &Ar[0];
    Amr  = NR;
    Bnr  = MR;
  }

  if ( alpha!=zero ) { 
    for ( k=0; k<kc; k++ ) {
      // Load colums/rows of A/B for current iteration
      A0 = _mm256_loadu_ps(&Aptr[baseA+0]);
      B0 = _mm256_loadu_ps(&Bptr[baseB+0]);

      B0 = _mm256_broadcast_ss(&Bptr[baseB + 0]);
      C00 = _mm256_fmadd_ps(A0, B0, C00);

      B1 = _mm256_broadcast_ss(&Bptr[baseB + 1]);
      C01 = _mm256_fmadd_ps(A0, B1, C01);
        
      B0 = _mm256_broadcast_ss(&Bptr[baseB + 2]);
      C02 = _mm256_fmadd_ps(A0, B0, C02);
      
      B1 = _mm256_broadcast_ss(&Bptr[baseB + 3]);
      C03 = _mm256_fmadd_ps(A0, B1, C03);
      
      B0 = _mm256_broadcast_ss(&Bptr[baseB + 4]);
      C04 = _mm256_fmadd_ps(A0, B0, C04);
      
      B1 = _mm256_broadcast_ss(&Bptr[baseB + 5]);
      C05 = _mm256_fmadd_ps(A0, B1, C05);
      
      B0 = _mm256_broadcast_ss(&Bptr[baseB + 6]);
      C06 = _mm256_fmadd_ps(A0, B0, C06);
      
      B1 = _mm256_broadcast_ss(&Bptr[baseB + 7]);
      C07 = _mm256_fmadd_ps(A0, B1, C07);
      
      baseA = baseA+Amr;
      baseB = baseB+Bnr;
    }
  }

   _mm256_storeu_ps(&Ctref(0, 0), C00);
   _mm256_storeu_ps(&Ctref(0, 1), C01);
   _mm256_storeu_ps(&Ctref(0, 2), C02);
   _mm256_storeu_ps(&Ctref(0, 3), C03);
   _mm256_storeu_ps(&Ctref(0, 4), C04);
   _mm256_storeu_ps(&Ctref(0, 5), C05);
   _mm256_storeu_ps(&Ctref(0, 6), C06);
   _mm256_storeu_ps(&Ctref(0, 7), C07);
   
   if ( beta!=zero ) {
     for ( j=0; j<nr; j++ )
       for ( i=0; i<mr; i++ )
         Ccol(i,j) = beta*Ccol(i,j) + Ctcol(i,j);
   } else {
     for ( j=0; j<nr; j++ )
       for ( i=0; i<mr; i++ )
         Ccol(i,j) = Ctcol(i,j);
   }
}

void gemm_microkernel_Cresident_AMD_avx256_8x14_fp32( char orderC, int mr, int nr, int kc, float alpha, 
		                                           float *Ar, float *Br, float beta, float *C, int ldC ) {

  int i, j, k, baseB = 0, baseA = 0, ldCt = MR, Amr, Bnr;
  __m256 C00, C01, C02, C03, C04, C05, C06, C07, C08, C09, C010, C011, C012, C013;
  __m256 A0, B0;

  float   zero = 0.0, *Aptr, *Bptr, Ctmp[MR*NR];

  if ( kc==0 )
    return;

  C00  = _mm256_set1_ps ( 0 ); 
  C01  = _mm256_set1_ps ( 0 ); 
  C02  = _mm256_set1_ps ( 0 ); 
  C03  = _mm256_set1_ps ( 0 ); 
  C04  = _mm256_set1_ps ( 0 ); 
  C05  = _mm256_set1_ps ( 0 ); 
  C06  = _mm256_set1_ps ( 0 ); 
  C07  = _mm256_set1_ps ( 0 ); 
  C08  = _mm256_set1_ps ( 0 ); 
  C09  = _mm256_set1_ps ( 0 ); 
  C010 = _mm256_set1_ps ( 0 ); 
  C011 = _mm256_set1_ps ( 0 ); 
  C012 = _mm256_set1_ps ( 0 ); 
  C013 = _mm256_set1_ps ( 0 ); 
  
  Aptr = &Ar[0];
  Bptr = &Br[0];
  Amr  = MR;
  Bnr  = NR;

  if ( alpha!=zero ) { 
    for ( k=0; k<kc; k++ ) {
      // Load colums/rows of A/B for current iteration
      A0 = _mm256_loadu_ps(&Aptr[baseA+0]);

      B0 = _mm256_broadcast_ss(&Bptr[baseB + 0]);
      C00 = _mm256_fmadd_ps(A0, B0, C00);

      B0 = _mm256_broadcast_ss(&Bptr[baseB + 1]);
      C01 = _mm256_fmadd_ps(A0, B0, C01);
        
      B0 = _mm256_broadcast_ss(&Bptr[baseB + 2]);
      C02 = _mm256_fmadd_ps(A0, B0, C02);
      
      B0 = _mm256_broadcast_ss(&Bptr[baseB + 3]);
      C03 = _mm256_fmadd_ps(A0, B0, C03);
      
      B0 = _mm256_broadcast_ss(&Bptr[baseB + 4]);
      C04 = _mm256_fmadd_ps(A0, B0, C04);
      
      B0 = _mm256_broadcast_ss(&Bptr[baseB + 5]);
      C05 = _mm256_fmadd_ps(A0, B0, C05);
      
      B0 = _mm256_broadcast_ss(&Bptr[baseB + 6]);
      C06 = _mm256_fmadd_ps(A0, B0, C06);
      
      B0 = _mm256_broadcast_ss(&Bptr[baseB + 7]);
      C07 = _mm256_fmadd_ps(A0, B0, C07);
      
      B0 = _mm256_broadcast_ss(&Bptr[baseB + 8]);
      C08 = _mm256_fmadd_ps(A0, B0, C08);
      
      B0 = _mm256_broadcast_ss(&Bptr[baseB + 9]);
      C09 = _mm256_fmadd_ps(A0, B0, C09);
      
      B0 = _mm256_broadcast_ss(&Bptr[baseB + 10]);
      C010 = _mm256_fmadd_ps(A0, B0, C010);
      
      B0 = _mm256_broadcast_ss(&Bptr[baseB + 11]);
      C011 = _mm256_fmadd_ps(A0, B0, C011);
      
      B0 = _mm256_broadcast_ss(&Bptr[baseB + 12]);
      C012 = _mm256_fmadd_ps(A0, B0, C012);
      
      B0 = _mm256_broadcast_ss(&Bptr[baseB + 13]);
      C013 = _mm256_fmadd_ps(A0, B0, C013);
      
      baseA = baseA+Amr;
      baseB = baseB+Bnr;
    }
  }

  if ((mr < MR) || (nr < NR)) {
   _mm256_storeu_ps(&Ctref(0, 0), C00);
   _mm256_storeu_ps(&Ctref(0, 1), C01);
   _mm256_storeu_ps(&Ctref(0, 2), C02);
   _mm256_storeu_ps(&Ctref(0, 3), C03);
   _mm256_storeu_ps(&Ctref(0, 4), C04);
   _mm256_storeu_ps(&Ctref(0, 5), C05);
   _mm256_storeu_ps(&Ctref(0, 6), C06);
   _mm256_storeu_ps(&Ctref(0, 7), C07);
   _mm256_storeu_ps(&Ctref(0, 8), C08);
   _mm256_storeu_ps(&Ctref(0, 9), C09);
   _mm256_storeu_ps(&Ctref(0, 10), C010);
   _mm256_storeu_ps(&Ctref(0, 11), C011);
   _mm256_storeu_ps(&Ctref(0, 12), C012);
   _mm256_storeu_ps(&Ctref(0, 13), C013);
   
   if ( beta!=zero ) {
     for ( j=0; j<nr; j++ )
       for ( i=0; i<mr; i++ )
         Ccol(i,j) = beta*Ccol(i,j) + Ctcol(i,j);
   } else {
     for ( j=0; j<nr; j++ )
       for ( i=0; i<mr; i++ )
         Ccol(i,j) = Ctcol(i,j);
   }
  } else {
    if ( beta!=zero ) {
      C00  += beta * _mm256_loadu_ps(&Ccol(0,0));
      C01  += beta * _mm256_loadu_ps(&Ccol(0,1));
      C02  += beta * _mm256_loadu_ps(&Ccol(0,2));
      C03  += beta * _mm256_loadu_ps(&Ccol(0,3));
      C04  += beta * _mm256_loadu_ps(&Ccol(0,4));
      C05  += beta * _mm256_loadu_ps(&Ccol(0,5));
      C06  += beta * _mm256_loadu_ps(&Ccol(0,6));
      C07  += beta * _mm256_loadu_ps(&Ccol(0,7));
      C08  += beta * _mm256_loadu_ps(&Ccol(0,8));
      C09  += beta * _mm256_loadu_ps(&Ccol(0,9));
      C010 += beta * _mm256_loadu_ps(&Ccol(0,10));
      C011 += beta * _mm256_loadu_ps(&Ccol(0,11));
      C012 += beta * _mm256_loadu_ps(&Ccol(0,12));
      C013 += beta * _mm256_loadu_ps(&Ccol(0,13));
    }
    _mm256_storeu_ps(&Ccol(0,0),  C00);
    _mm256_storeu_ps(&Ccol(0,1),  C01);
    _mm256_storeu_ps(&Ccol(0,2),  C02);
    _mm256_storeu_ps(&Ccol(0,3),  C03);
    _mm256_storeu_ps(&Ccol(0,4),  C04);
    _mm256_storeu_ps(&Ccol(0,5),  C05);
    _mm256_storeu_ps(&Ccol(0,6),  C06);
    _mm256_storeu_ps(&Ccol(0,7),  C07);
    _mm256_storeu_ps(&Ccol(0,8),  C08);
    _mm256_storeu_ps(&Ccol(0,9),  C09);
    _mm256_storeu_ps(&Ccol(0,10), C010);
    _mm256_storeu_ps(&Ccol(0,11), C011);
    _mm256_storeu_ps(&Ccol(0,12), C012);
    _mm256_storeu_ps(&Ccol(0,13), C013);
  }
}

void gemm_microkernel_Cresident_AMD_avx256_16x6_optimum_fp32( char orderC, int mr, int nr, int kc, float alpha, 
		                                              float *Ar, float *Br, float beta, float *C, int ldC ) {

  int i, j, k, baseB = 0, baseA = 0, ldCt = MR, Amr, Bnr, Amr_unroll, Bnr_unroll;
  __m256 C00, C01, C02, C03, C04, C05;
  __m256 C10, C11, C12, C13, C14, C15;
  
  __m256 A0, A1, B0, B1;

  float   zero = 0.0, *Aptr, *Bptr, Ctmp[MR*NR];

  //int kc_iter, kc_left;
  if ( kc==0 )
    return;


  if ( beta!=zero ) {
    C00  = _mm256_loadu_ps(&Ccol(0,0)); 
    C01  = _mm256_loadu_ps(&Ccol(0,1));
    C02  = _mm256_loadu_ps(&Ccol(0,2));
    C03  = _mm256_loadu_ps(&Ccol(0,3));
    C04  = _mm256_loadu_ps(&Ccol(0,4));
    C05  = _mm256_loadu_ps(&Ccol(0,5));
  
    C10  = _mm256_loadu_ps(&Ccol(8,0));
    C11  = _mm256_loadu_ps(&Ccol(8,1));
    C12  = _mm256_loadu_ps(&Ccol(8,2));
    C13  = _mm256_loadu_ps(&Ccol(8,3));
    C14  = _mm256_loadu_ps(&Ccol(8,4));
    C15  = _mm256_loadu_ps(&Ccol(8,5));
  } else {
    C00  = _mm256_set1_ps ( 0 ); 
    C01  = _mm256_set1_ps ( 0 ); 
    C02  = _mm256_set1_ps ( 0 ); 
    C03  = _mm256_set1_ps ( 0 ); 
    C04  = _mm256_set1_ps ( 0 ); 
    C05  = _mm256_set1_ps ( 0 ); 
  
    C10  = _mm256_set1_ps ( 0 ); 
    C11  = _mm256_set1_ps ( 0 ); 
    C12  = _mm256_set1_ps ( 0 ); 
    C13  = _mm256_set1_ps ( 0 ); 
    C14  = _mm256_set1_ps ( 0 ); 
    C15  = _mm256_set1_ps ( 0 ); 
  }

  Aptr = &Ar[0];
  Bptr = &Br[0];

  if ( alpha!=zero ) {
    if (mr == MR && nr == NR) {
      //kc_iter = kc / 2;
      //kc_left = kc % 2;
      Amr_unroll = MR * 2;
      Bnr_unroll = NR * 2;
      for ( k=0; k < kc; k += 2 ) {
        A0 = _mm256_loadu_ps(&Aptr[baseA + 0]);
        A1 = _mm256_loadu_ps(&Aptr[baseA + 8]);
 
	B0 = _mm256_broadcast_ss(&Bptr[baseB + 0]);
        C00 = _mm256_fmadd_ps(A0, B0, C00);
        C10 = _mm256_fmadd_ps(A1, B0, C10);

        B1 = _mm256_broadcast_ss(&Bptr[baseB + 1]);
        C01 = _mm256_fmadd_ps(A0, B1, C01);
        C11 = _mm256_fmadd_ps(A1, B1, C11);
        
	B0 = _mm256_broadcast_ss(&Bptr[baseB + 2]);
        C02 = _mm256_fmadd_ps(A0, B0, C02);
        C12 = _mm256_fmadd_ps(A1, B0, C12);
      
        B1 = _mm256_broadcast_ss(&Bptr[baseB + 3]);
        C03 = _mm256_fmadd_ps(A0, B1, C03);
        C13 = _mm256_fmadd_ps(A1, B1, C13);
      
        B0 = _mm256_broadcast_ss(&Bptr[baseB + 4]);
        C04 = _mm256_fmadd_ps(A0, B0, C04);
        C14 = _mm256_fmadd_ps(A1, B0, C14);
      
        B1 = _mm256_broadcast_ss(&Bptr[baseB + 5]);
        C05 = _mm256_fmadd_ps(A0, B1, C05);
        C15 = _mm256_fmadd_ps(A1, B1, C15);
      
	A0 = _mm256_loadu_ps(&Aptr[baseA + 16]);
        A1 = _mm256_loadu_ps(&Aptr[baseA + 24]);
 
	B0 = _mm256_broadcast_ss(&Bptr[baseB + 6]);
        C00 = _mm256_fmadd_ps(A0, B0, C00);
        C10 = _mm256_fmadd_ps(A1, B0, C10);

        B1 = _mm256_broadcast_ss(&Bptr[baseB + 7]);
        C01 = _mm256_fmadd_ps(A0, B1, C01);
        C11 = _mm256_fmadd_ps(A1, B1, C11);
        
	B0 = _mm256_broadcast_ss(&Bptr[baseB + 8]);
        C02 = _mm256_fmadd_ps(A0, B0, C02);
        C12 = _mm256_fmadd_ps(A1, B0, C12);
      
        B1 = _mm256_broadcast_ss(&Bptr[baseB + 9]);
        C03 = _mm256_fmadd_ps(A0, B1, C03);
        C13 = _mm256_fmadd_ps(A1, B1, C13);
      
        B0 = _mm256_broadcast_ss(&Bptr[baseB + 10]);
        C04 = _mm256_fmadd_ps(A0, B0, C04);
        C14 = _mm256_fmadd_ps(A1, B0, C14);
      
        B1 = _mm256_broadcast_ss(&Bptr[baseB + 11]);
        C05 = _mm256_fmadd_ps(A0, B1, C05);
        C15 = _mm256_fmadd_ps(A1, B1, C15);

        baseA = baseA+Amr_unroll;
        baseB = baseB+Bnr_unroll;
      }

      if (kc % 2 > 0) {
        A0 = _mm256_loadu_ps(&Aptr[baseA + 0]);
        A1 = _mm256_loadu_ps(&Aptr[baseA + 8]);
 
	B0 = _mm256_broadcast_ss(&Bptr[baseB + 0]);
        C00 = _mm256_fmadd_ps(A0, B0, C00);
        C10 = _mm256_fmadd_ps(A1, B0, C10);

        B1 = _mm256_broadcast_ss(&Bptr[baseB + 1]);
        C01 = _mm256_fmadd_ps(A0, B1, C01);
        C11 = _mm256_fmadd_ps(A1, B1, C11);
        
	B0 = _mm256_broadcast_ss(&Bptr[baseB + 2]);
        C02 = _mm256_fmadd_ps(A0, B0, C02);
        C12 = _mm256_fmadd_ps(A1, B0, C12);
      
        B1 = _mm256_broadcast_ss(&Bptr[baseB + 3]);
        C03 = _mm256_fmadd_ps(A0, B1, C03);
        C13 = _mm256_fmadd_ps(A1, B1, C13);
      
        B0 = _mm256_broadcast_ss(&Bptr[baseB + 4]);
        C04 = _mm256_fmadd_ps(A0, B0, C04);
        C14 = _mm256_fmadd_ps(A1, B0, C14);
      
        B1 = _mm256_broadcast_ss(&Bptr[baseB + 5]);
        C05 = _mm256_fmadd_ps(A0, B1, C05);
        C15 = _mm256_fmadd_ps(A1, B1, C15); 
      }

  } else {
    Amr  = MR;
    Bnr  = NR;
    if ((mr <= 8) && (nr <= 2)) {
      for ( k=0; k < kc; k++ ) {
        A0 = _mm256_loadu_ps(&Aptr[baseA + 0]);

        B0 = _mm256_broadcast_ss(&Bptr[baseB + 0]);
        C00 = _mm256_fmadd_ps(A0, B0, C00);

        B0 = _mm256_broadcast_ss(&Bptr[baseB + 1]);
        C01 = _mm256_fmadd_ps(A0, B0, C01);
      
        baseA = baseA+Amr;
        baseB = baseB+Bnr;
      }
    } else if ((mr <= 8) && (nr <= 4)) {
      for ( k=0; k < kc; k++ ) {
        A0 = _mm256_loadu_ps(&Aptr[baseA + 0]);

        B0 = _mm256_broadcast_ss(&Bptr[baseB + 0]);
        C00 = _mm256_fmadd_ps(A0, B0, C00);

        B0 = _mm256_broadcast_ss(&Bptr[baseB + 1]);
        C01 = _mm256_fmadd_ps(A0, B0, C01);
        
	B0 = _mm256_broadcast_ss(&Bptr[baseB + 2]);
        C02 = _mm256_fmadd_ps(A0, B0, C02);

        B0 = _mm256_broadcast_ss(&Bptr[baseB + 3]);
        C03 = _mm256_fmadd_ps(A0, B0, C03);
      
        baseA = baseA+Amr;
        baseB = baseB+Bnr;
      } 
    } else if ((mr <= 16) && (nr <= 2)) {
      for ( k=0; k < kc; k++ ) {
        A0 = _mm256_loadu_ps(&Aptr[baseA + 0]);
        A1 = _mm256_loadu_ps(&Aptr[baseA + 8]);

        B0 = _mm256_broadcast_ss(&Bptr[baseB + 0]);
        C00 = _mm256_fmadd_ps(A0, B0, C00);
        C10 = _mm256_fmadd_ps(A1, B0, C10);

        B0 = _mm256_broadcast_ss(&Bptr[baseB + 1]);
        C01 = _mm256_fmadd_ps(A0, B0, C01);
        C11 = _mm256_fmadd_ps(A1, B0, C11);
      
        baseA = baseA+Amr;
        baseB = baseB+Bnr;
      }
    } else if ((mr <= 16) && (nr <= 4)) {
      for ( k=0; k < kc; k++ ) {
        A0 = _mm256_loadu_ps(&Aptr[baseA + 0]);
        A1 = _mm256_loadu_ps(&Aptr[baseA + 8]);

        B0 = _mm256_broadcast_ss(&Bptr[baseB + 0]);
        C00 = _mm256_fmadd_ps(A0, B0, C00);
        C10 = _mm256_fmadd_ps(A1, B0, C10);

        B0 = _mm256_broadcast_ss(&Bptr[baseB + 1]);
        C01 = _mm256_fmadd_ps(A0, B0, C01);
        C11 = _mm256_fmadd_ps(A1, B0, C11);
        
	B0 = _mm256_broadcast_ss(&Bptr[baseB + 2]);
        C02 = _mm256_fmadd_ps(A0, B0, C02);
        C12 = _mm256_fmadd_ps(A1, B0, C12);

        B0 = _mm256_broadcast_ss(&Bptr[baseB + 3]);
        C03 = _mm256_fmadd_ps(A0, B0, C03);
        C13 = _mm256_fmadd_ps(A1, B0, C13);
      
        baseA = baseA+Amr;
        baseB = baseB+Bnr;
      }
    } else {
        for ( k=0; k < kc; k++ ) {
          // Load colums/rows of A/B for current iteration
          A0 = _mm256_loadu_ps(&Aptr[baseA + 0]);
          A1 = _mm256_loadu_ps(&Aptr[baseA + 8]);

          B0 = _mm256_broadcast_ss(&Bptr[baseB + 0]);
          C00 = _mm256_fmadd_ps(A0, B0, C00);
          C10 = _mm256_fmadd_ps(A1, B0, C10);

          B0 = _mm256_broadcast_ss(&Bptr[baseB + 1]);
          C01 = _mm256_fmadd_ps(A0, B0, C01);
          C11 = _mm256_fmadd_ps(A1, B0, C11);
      
          B0 = _mm256_broadcast_ss(&Bptr[baseB + 2]);
          C02 = _mm256_fmadd_ps(A0, B0, C02);
          C12 = _mm256_fmadd_ps(A1, B0, C12);
      
          B0 = _mm256_broadcast_ss(&Bptr[baseB + 3]);
          C03 = _mm256_fmadd_ps(A0, B0, C03);
          C13 = _mm256_fmadd_ps(A1, B0, C13);
      
          B0 = _mm256_broadcast_ss(&Bptr[baseB + 4]);
          C04 = _mm256_fmadd_ps(A0, B0, C04);
          C14 = _mm256_fmadd_ps(A1, B0, C14);
      
          B0 = _mm256_broadcast_ss(&Bptr[baseB + 5]);
          C05 = _mm256_fmadd_ps(A0, B0, C05);
          C15 = _mm256_fmadd_ps(A1, B0, C15);
      
          baseA = baseA+Amr;
          baseB = baseB+Bnr;
        }
      }
    }
  }

  if ((mr == MR) && (nr == NR)) {
    _mm256_storeu_ps(&Ccol(0,0),  C00);
    _mm256_storeu_ps(&Ccol(0,1),  C01);
    _mm256_storeu_ps(&Ccol(0,2),  C02);
    _mm256_storeu_ps(&Ccol(0,3),  C03);
    _mm256_storeu_ps(&Ccol(0,4),  C04);
    _mm256_storeu_ps(&Ccol(0,5),  C05);
    
    _mm256_storeu_ps(&Ccol(8,0),  C10);
    _mm256_storeu_ps(&Ccol(8,1),  C11);
    _mm256_storeu_ps(&Ccol(8,2),  C12);
    _mm256_storeu_ps(&Ccol(8,3),  C13);
    _mm256_storeu_ps(&Ccol(8,4),  C14);
    _mm256_storeu_ps(&Ccol(8,5),  C15);
   
  } else {
   _mm256_storeu_ps(&Ctref(0, 0), C00);
   _mm256_storeu_ps(&Ctref(0, 1), C01);
   _mm256_storeu_ps(&Ctref(0, 2), C02);
   _mm256_storeu_ps(&Ctref(0, 3), C03);
   _mm256_storeu_ps(&Ctref(0, 4), C04);
   _mm256_storeu_ps(&Ctref(0, 5), C05);

   _mm256_storeu_ps(&Ctref(8, 0), C10);
   _mm256_storeu_ps(&Ctref(8, 1), C11);
   _mm256_storeu_ps(&Ctref(8, 2), C12);
   _mm256_storeu_ps(&Ctref(8, 3), C13);
   _mm256_storeu_ps(&Ctref(8, 4), C14);
   _mm256_storeu_ps(&Ctref(8, 5), C15);

     for ( j=0; j<nr; j++ )
       for ( i=0; i<mr; i++ )
         Ccol(i,j) = Ctcol(i,j);
  }
}

//-------------------------------------------------------------------------------------------------//
//*************************************************************************************************//
//-------------------------------------------------------------------------------------------------//

void gemm_microkernel_Cresident_AMD_avx256_2vx6_fp32( char orderC, int mr, int nr, int kc, float alpha, 
		                                      float *Ar, float *Br, float beta, float *C, int ldC ) {

  int i, j, k, baseB = 0, baseA = 0, ldCt = MR, Amr, Bnr;
  __m256 C00, C01, C02, C03, C04, C05;
  __m256 C10, C11, C12, C13, C14, C15;
  
  __m256 A0, A1, B0;

  float   zero = 0.0, *Aptr, *Bptr, Ctmp[MR*NR];

  if ( kc==0 )
    return;

  if ( beta!=zero ) {
    C00  = _mm256_loadu_ps(&Ccol(0,0)); 
    C01  = _mm256_loadu_ps(&Ccol(0,1));
    C02  = _mm256_loadu_ps(&Ccol(0,2));
    C03  = _mm256_loadu_ps(&Ccol(0,3));
    C04  = _mm256_loadu_ps(&Ccol(0,4));
    C05  = _mm256_loadu_ps(&Ccol(0,5));
  
    C10  = _mm256_loadu_ps(&Ccol(8,0));
    C11  = _mm256_loadu_ps(&Ccol(8,1));
    C12  = _mm256_loadu_ps(&Ccol(8,2));
    C13  = _mm256_loadu_ps(&Ccol(8,3));
    C14  = _mm256_loadu_ps(&Ccol(8,4));
    C15  = _mm256_loadu_ps(&Ccol(8,5));
  } else {
    C00  = _mm256_set1_ps ( 0 ); 
    C01  = _mm256_set1_ps ( 0 ); 
    C02  = _mm256_set1_ps ( 0 ); 
    C03  = _mm256_set1_ps ( 0 ); 
    C04  = _mm256_set1_ps ( 0 ); 
    C05  = _mm256_set1_ps ( 0 ); 
  
    C10  = _mm256_set1_ps ( 0 ); 
    C11  = _mm256_set1_ps ( 0 ); 
    C12  = _mm256_set1_ps ( 0 ); 
    C13  = _mm256_set1_ps ( 0 ); 
    C14  = _mm256_set1_ps ( 0 ); 
    C15  = _mm256_set1_ps ( 0 ); 
  }

  if ( orderC=='C' ) {
    Aptr = &Ar[0];
    Bptr = &Br[0];
    Amr  = MR;
    Bnr  = NR;
  } else {
    Aptr = &Br[0];
    Bptr = &Ar[0];
    Amr  = NR;
    Bnr  = MR;
  }


  if ( alpha!=zero ) {
    //if (mr == MR && nr == NR) 
      for ( k=0; k < kc; k++ ) {
        // Load colums/rows of A/B for current iteration
        A0 = _mm256_loadu_ps(&Aptr[baseA + 0]);
        A1 = _mm256_loadu_ps(&Aptr[baseA + 8]);

        B0 = _mm256_broadcast_ss(&Bptr[baseB + 0]);
        C00 = _mm256_fmadd_ps(A0, B0, C00);
        C10 = _mm256_fmadd_ps(A1, B0, C10);

        B0 = _mm256_broadcast_ss(&Bptr[baseB + 1]);
        C01 = _mm256_fmadd_ps(A0, B0, C01);
        C11 = _mm256_fmadd_ps(A1, B0, C11);
      
        B0 = _mm256_broadcast_ss(&Bptr[baseB + 2]);
        C02 = _mm256_fmadd_ps(A0, B0, C02);
        C12 = _mm256_fmadd_ps(A1, B0, C12);
      
        B0 = _mm256_broadcast_ss(&Bptr[baseB + 3]);
        C03 = _mm256_fmadd_ps(A0, B0, C03);
        C13 = _mm256_fmadd_ps(A1, B0, C13);
      
        B0 = _mm256_broadcast_ss(&Bptr[baseB + 4]);
        C04 = _mm256_fmadd_ps(A0, B0, C04);
        C14 = _mm256_fmadd_ps(A1, B0, C14);
      
        B0 = _mm256_broadcast_ss(&Bptr[baseB + 5]);
        C05 = _mm256_fmadd_ps(A0, B0, C05);
        C15 = _mm256_fmadd_ps(A1, B0, C15);
         
        baseA = baseA+Amr;
        baseB = baseB+Bnr;
      }
  }

  if ((mr < MR) || (nr < NR)) {
   
   _mm256_storeu_ps(&Ctref(0, 0), C00);
   _mm256_storeu_ps(&Ctref(0, 1), C01);
   _mm256_storeu_ps(&Ctref(0, 2), C02);
   _mm256_storeu_ps(&Ctref(0, 3), C03);
   _mm256_storeu_ps(&Ctref(0, 4), C04);
   _mm256_storeu_ps(&Ctref(0, 5), C05);

   _mm256_storeu_ps(&Ctref(8, 0), C10);
   _mm256_storeu_ps(&Ctref(8, 1), C11);
   _mm256_storeu_ps(&Ctref(8, 2), C12);
   _mm256_storeu_ps(&Ctref(8, 3), C13);
   _mm256_storeu_ps(&Ctref(8, 4), C14);
   _mm256_storeu_ps(&Ctref(8, 5), C15);

     for ( j=0; j<nr; j++ )
       for ( i=0; i<mr; i++ )
         Ccol(i,j) = Ctcol(i,j);
  
  } else {
    _mm256_storeu_ps(&Ccol(0,0),  C00);
    _mm256_storeu_ps(&Ccol(0,1),  C01);
    _mm256_storeu_ps(&Ccol(0,2),  C02);
    _mm256_storeu_ps(&Ccol(0,3),  C03);
    _mm256_storeu_ps(&Ccol(0,4),  C04);
    _mm256_storeu_ps(&Ccol(0,5),  C05);
    
    _mm256_storeu_ps(&Ccol(8,0),  C10);
    _mm256_storeu_ps(&Ccol(8,1),  C11);
    _mm256_storeu_ps(&Ccol(8,2),  C12);
    _mm256_storeu_ps(&Ccol(8,3),  C13);
    _mm256_storeu_ps(&Ccol(8,4),  C14);
    _mm256_storeu_ps(&Ccol(8,5),  C15);
  }
}

void gemm_microkernel_Cresident_AMD_avx256_6x2v_fp32( char orderC, int mr, int nr, int kc, float alpha, 
		                                      float *Ar, float *Br, float beta, float *C, int ldC ) {

  int i, j, k, baseB = 0, baseA = 0, ldCt = NR, Amr, Bnr;
  __m256 C00, C01, C02, C03, C04, C05;
  __m256 C10, C11, C12, C13, C14, C15;
  
  __m256 B0, B1, A0;

  float   zero = 0.0, *Aptr, *Bptr, Ctmp[MR*NR];

  if ( kc==0 )
    return;

  if ( beta!=zero ) {
    
    C00 = _mm256_loadu_ps(&Crow(0, 0));
    C01 = _mm256_loadu_ps(&Crow(1, 0));
    C02 = _mm256_loadu_ps(&Crow(2, 0));
    C03 = _mm256_loadu_ps(&Crow(3, 0));
    C04 = _mm256_loadu_ps(&Crow(4, 0));
    C05 = _mm256_loadu_ps(&Crow(5, 0));

    C10 = _mm256_loadu_ps(&Crow(0, 8));
    C11 = _mm256_loadu_ps(&Crow(1, 8));
    C12 = _mm256_loadu_ps(&Crow(2, 8));
    C13 = _mm256_loadu_ps(&Crow(3, 8));
    C14 = _mm256_loadu_ps(&Crow(4, 8));
    C15 = _mm256_loadu_ps(&Crow(5, 8));

  } else {
    C00  = _mm256_set1_ps ( 0 ); 
    C01  = _mm256_set1_ps ( 0 ); 
    C02  = _mm256_set1_ps ( 0 ); 
    C03  = _mm256_set1_ps ( 0 ); 
    C04  = _mm256_set1_ps ( 0 ); 
    C05  = _mm256_set1_ps ( 0 ); 
  
    C10  = _mm256_set1_ps ( 0 ); 
    C11  = _mm256_set1_ps ( 0 ); 
    C12  = _mm256_set1_ps ( 0 ); 
    C13  = _mm256_set1_ps ( 0 ); 
    C14  = _mm256_set1_ps ( 0 ); 
    C15  = _mm256_set1_ps ( 0 ); 
  }

  Aptr = &Ar[0];
  Bptr = &Br[0];
  Amr  = MR;
  Bnr  = NR;

  if ( alpha!=zero ) {
    //if (mr == MR && nr == NR) 
      for ( k=0; k < kc; k++ ) {
        // Load colums/rows of A/B for current iteration
        B0 = _mm256_loadu_ps(&Bptr[baseB + 0]);
        B1 = _mm256_loadu_ps(&Bptr[baseB + 8]);
        
        A0 = _mm256_broadcast_ss(&Aptr[baseA + 0]);
        C00 = _mm256_fmadd_ps(B0, A0, C00);
        C10 = _mm256_fmadd_ps(B1, A0, C10);
        
        A0 = _mm256_broadcast_ss(&Aptr[baseA + 1]);
        C01 = _mm256_fmadd_ps(B0, A0, C01);
        C11 = _mm256_fmadd_ps(B1, A0, C11);
      
        A0 = _mm256_broadcast_ss(&Aptr[baseA + 2]);
        C02 = _mm256_fmadd_ps(B0, A0, C02);
        C12 = _mm256_fmadd_ps(B1, A0, C12);

        A0 = _mm256_broadcast_ss(&Aptr[baseA + 3]);
        C03 = _mm256_fmadd_ps(B0, A0, C03);
        C13 = _mm256_fmadd_ps(B1, A0, C13);
      
        A0 = _mm256_broadcast_ss(&Aptr[baseA + 4]);
        C04 = _mm256_fmadd_ps(B0, A0, C04);
        C14 = _mm256_fmadd_ps(B1, A0, C14);
      
        A0 = _mm256_broadcast_ss(&Aptr[baseA + 5]);
        C05 = _mm256_fmadd_ps(B0, A0, C05);
        C15 = _mm256_fmadd_ps(B1, A0, C15);
         
        baseA = baseA + Amr;
        baseB = baseB + Bnr;
      }
  }

  if ((mr < MR) || (nr < NR)) {
   
    _mm256_storeu_ps(&Ctrow(0, 0), C00); 
    _mm256_storeu_ps(&Ctrow(1, 0), C01); 
    _mm256_storeu_ps(&Ctrow(2, 0), C02);
    _mm256_storeu_ps(&Ctrow(3, 0), C03);
    _mm256_storeu_ps(&Ctrow(4, 0), C04);
    _mm256_storeu_ps(&Ctrow(5, 0), C05);

    _mm256_storeu_ps(&Ctrow(0, 8), C10); 
    _mm256_storeu_ps(&Ctrow(1, 8), C11); 
    _mm256_storeu_ps(&Ctrow(2, 8), C12);
    _mm256_storeu_ps(&Ctrow(3, 8), C13);
    _mm256_storeu_ps(&Ctrow(4, 8), C14);
    _mm256_storeu_ps(&Ctrow(5, 8), C15);

     for ( j=0; j<nr; j++ )
       for ( i=0; i<mr; i++ )
         Crow(i,j) = Ctrow(i,j);
  
  } else {

    _mm256_storeu_ps(&Crow(0, 0), C00);
    _mm256_storeu_ps(&Crow(1, 0), C01);
    _mm256_storeu_ps(&Crow(2, 0), C02);
    _mm256_storeu_ps(&Crow(3, 0), C03);
    _mm256_storeu_ps(&Crow(4, 0), C04);
    _mm256_storeu_ps(&Crow(5, 0), C05);
    
    _mm256_storeu_ps(&Crow(0, 8), C10);
    _mm256_storeu_ps(&Crow(1, 8), C11);
    _mm256_storeu_ps(&Crow(2, 8), C12);
    _mm256_storeu_ps(&Crow(3, 8), C13);
    _mm256_storeu_ps(&Crow(4, 8), C14);
    _mm256_storeu_ps(&Crow(5, 8), C15);
     
  }
}



void gemm_microkernel_Cresident_AMD_avx256_3vx4_fp32( char orderC, int mr, int nr, int kc, float alpha, 
		                                      float *Ar, float *Br, float beta, float *C, int ldC ) {

  int i, j, k, baseB = 0, baseA = 0, ldCt = MR, Amr, Bnr;
  __m256 C00, C01, C02, C03,
         C10, C11, C12, C13,
         C20, C21, C22, C23;
  
  __m256 A0, A1, A2, B0;

  float   zero = 0.0, *Aptr, *Bptr, Ctmp[MR*NR];

  if ( kc==0 )
    return;

  if ( beta!=zero ) {
    C00  = _mm256_loadu_ps(&Ccol(0,0)); 
    C01  = _mm256_loadu_ps(&Ccol(0,1));
    C02  = _mm256_loadu_ps(&Ccol(0,2));
    C03  = _mm256_loadu_ps(&Ccol(0,3));
  
    C10  = _mm256_loadu_ps(&Ccol(8,0));
    C11  = _mm256_loadu_ps(&Ccol(8,1));
    C12  = _mm256_loadu_ps(&Ccol(8,2));
    C13  = _mm256_loadu_ps(&Ccol(8,3));
    
    C20  = _mm256_loadu_ps(&Ccol(16,0));
    C21  = _mm256_loadu_ps(&Ccol(16,1));
    C22  = _mm256_loadu_ps(&Ccol(16,2));
    C23  = _mm256_loadu_ps(&Ccol(16,3));
  } else {
    C00  = _mm256_set1_ps ( 0 ); 
    C01  = _mm256_set1_ps ( 0 ); 
    C02  = _mm256_set1_ps ( 0 ); 
    C03  = _mm256_set1_ps ( 0 ); 
  
    C10  = _mm256_set1_ps ( 0 ); 
    C11  = _mm256_set1_ps ( 0 ); 
    C12  = _mm256_set1_ps ( 0 ); 
    C13  = _mm256_set1_ps ( 0 ); 
    
    C20  = _mm256_set1_ps ( 0 ); 
    C21  = _mm256_set1_ps ( 0 ); 
    C22  = _mm256_set1_ps ( 0 ); 
    C23  = _mm256_set1_ps ( 0 ); 
  }

  Aptr = &Ar[0];
  Bptr = &Br[0];
  Amr  = MR;
  Bnr  = NR;

  if ( alpha!=zero ) {
    //if (mr == MR && nr == NR) 
      for ( k=0; k < kc; k++ ) {
        // Load colums/rows of A/B for current iteration
        A0 = _mm256_loadu_ps(&Aptr[baseA + 0]);
        A1 = _mm256_loadu_ps(&Aptr[baseA + 8]);
        A2 = _mm256_loadu_ps(&Aptr[baseA + 16]);

        B0 = _mm256_broadcast_ss(&Bptr[baseB + 0]);
        C00 = _mm256_fmadd_ps(A0, B0, C00);
        C10 = _mm256_fmadd_ps(A1, B0, C10);
        C20 = _mm256_fmadd_ps(A2, B0, C20);

        B0 = _mm256_broadcast_ss(&Bptr[baseB + 1]);
        C01 = _mm256_fmadd_ps(A0, B0, C01);
        C11 = _mm256_fmadd_ps(A1, B0, C11);
        C21 = _mm256_fmadd_ps(A2, B0, C21);
      
        B0 = _mm256_broadcast_ss(&Bptr[baseB + 2]);
        C02 = _mm256_fmadd_ps(A0, B0, C02);
        C12 = _mm256_fmadd_ps(A1, B0, C12);
        C22 = _mm256_fmadd_ps(A2, B0, C22);
      
        B0 = _mm256_broadcast_ss(&Bptr[baseB + 3]);
        C03 = _mm256_fmadd_ps(A0, B0, C03);
        C13 = _mm256_fmadd_ps(A1, B0, C13);
        C23 = _mm256_fmadd_ps(A2, B0, C23);
      
        baseA = baseA+Amr;
        baseB = baseB+Bnr;
      }
  }

  if ((mr < MR) || (nr < NR)) {
   
   _mm256_storeu_ps(&Ctref(0, 0), C00);
   _mm256_storeu_ps(&Ctref(0, 1), C01);
   _mm256_storeu_ps(&Ctref(0, 2), C02);
   _mm256_storeu_ps(&Ctref(0, 3), C03);

   _mm256_storeu_ps(&Ctref(8, 0), C10);
   _mm256_storeu_ps(&Ctref(8, 1), C11);
   _mm256_storeu_ps(&Ctref(8, 2), C12);
   _mm256_storeu_ps(&Ctref(8, 3), C13);

   _mm256_storeu_ps(&Ctref(16, 0), C20);
   _mm256_storeu_ps(&Ctref(16, 1), C21);
   _mm256_storeu_ps(&Ctref(16, 2), C22);
   _mm256_storeu_ps(&Ctref(16, 3), C23);
   
   for ( j=0; j<nr; j++ )
     for ( i=0; i<mr; i++ )
       Ccol(i,j) = Ctcol(i,j);
  
  } else {
    _mm256_storeu_ps(&Ccol(0,0),  C00);
    _mm256_storeu_ps(&Ccol(0,1),  C01);
    _mm256_storeu_ps(&Ccol(0,2),  C02);
    _mm256_storeu_ps(&Ccol(0,3),  C03);
    
    _mm256_storeu_ps(&Ccol(8,0),  C10);
    _mm256_storeu_ps(&Ccol(8,1),  C11);
    _mm256_storeu_ps(&Ccol(8,2),  C12);
    _mm256_storeu_ps(&Ccol(8,3),  C13);
    
    _mm256_storeu_ps(&Ccol(16,0),  C20);
    _mm256_storeu_ps(&Ccol(16,1),  C21);
    _mm256_storeu_ps(&Ccol(16,2),  C22);
    _mm256_storeu_ps(&Ccol(16,3),  C23);
  }
}

void gemm_microkernel_Cresident_AMD_avx256_4x3v_fp32( char orderC, int mr, int nr, int kc, float alpha, 
		                                      float *Ar, float *Br, float beta, float *C, int ldC ) {

  int i, j, k, baseB = 0, baseA = 0, ldCt = NR, Amr, Bnr;
  __m256 C00, C01, C02, C03,
         C10, C11, C12, C13, 
         C20, C21, C22, C23; 
  
  __m256 B0, B1, B2, A0;

  float   zero = 0.0, *Aptr, *Bptr, Ctmp[MR*NR];

  if ( kc==0 )
    return;

  if ( beta!=zero ) {
    
    C00 = _mm256_loadu_ps(&Crow(0, 0));
    C01 = _mm256_loadu_ps(&Crow(1, 0));
    C02 = _mm256_loadu_ps(&Crow(2, 0));
    C03 = _mm256_loadu_ps(&Crow(3, 0));

    C10 = _mm256_loadu_ps(&Crow(0, 8));
    C11 = _mm256_loadu_ps(&Crow(1, 8));
    C12 = _mm256_loadu_ps(&Crow(2, 8));
    C13 = _mm256_loadu_ps(&Crow(3, 8));

    C20 = _mm256_loadu_ps(&Crow(0, 16));
    C21 = _mm256_loadu_ps(&Crow(1, 16));
    C22 = _mm256_loadu_ps(&Crow(2, 16));
    C23 = _mm256_loadu_ps(&Crow(3, 16));
  } else {
    C00  = _mm256_set1_ps ( 0 ); 
    C01  = _mm256_set1_ps ( 0 ); 
    C02  = _mm256_set1_ps ( 0 ); 
    C03  = _mm256_set1_ps ( 0 ); 
  
    C10  = _mm256_set1_ps ( 0 ); 
    C11  = _mm256_set1_ps ( 0 ); 
    C12  = _mm256_set1_ps ( 0 ); 
    C13  = _mm256_set1_ps ( 0 ); 
    
    C20  = _mm256_set1_ps ( 0 ); 
    C21  = _mm256_set1_ps ( 0 ); 
    C22  = _mm256_set1_ps ( 0 ); 
    C23  = _mm256_set1_ps ( 0 ); 
  }

  Aptr = &Ar[0];
  Bptr = &Br[0];
  Amr  = MR;
  Bnr  = NR;

  if ( alpha!=zero ) {
      for ( k=0; k < kc; k++ ) {
        // Load colums/rows of A/B for current iteration
        B0 = _mm256_loadu_ps(&Bptr[baseB + 0]);
        B1 = _mm256_loadu_ps(&Bptr[baseB + 8]);
        B2 = _mm256_loadu_ps(&Bptr[baseB + 16]);
        
        A0 = _mm256_broadcast_ss(&Aptr[baseA + 0]);
        C00 = _mm256_fmadd_ps(B0, A0, C00);
        C10 = _mm256_fmadd_ps(B1, A0, C10);
        C20 = _mm256_fmadd_ps(B2, A0, C20);
        
        A0 = _mm256_broadcast_ss(&Aptr[baseA + 1]);
        C01 = _mm256_fmadd_ps(B0, A0, C01);
        C11 = _mm256_fmadd_ps(B1, A0, C11);
        C21 = _mm256_fmadd_ps(B2, A0, C21);
      
        A0 = _mm256_broadcast_ss(&Aptr[baseA + 2]);
        C02 = _mm256_fmadd_ps(B0, A0, C02);
        C12 = _mm256_fmadd_ps(B1, A0, C12);
        C22 = _mm256_fmadd_ps(B2, A0, C22);

        A0 = _mm256_broadcast_ss(&Aptr[baseA + 3]);
        C03 = _mm256_fmadd_ps(B0, A0, C03);
        C13 = _mm256_fmadd_ps(B1, A0, C13);
        C23 = _mm256_fmadd_ps(B2, A0, C23);
      
        baseA = baseA + Amr;
        baseB = baseB + Bnr;
      }
  }

  if ((mr < MR) || (nr < NR)) {
   
    _mm256_storeu_ps(&Ctrow(0, 0), C00); 
    _mm256_storeu_ps(&Ctrow(1, 0), C01); 
    _mm256_storeu_ps(&Ctrow(2, 0), C02);
    _mm256_storeu_ps(&Ctrow(3, 0), C03);

    _mm256_storeu_ps(&Ctrow(0, 8), C10); 
    _mm256_storeu_ps(&Ctrow(1, 8), C11); 
    _mm256_storeu_ps(&Ctrow(2, 8), C12);
    _mm256_storeu_ps(&Ctrow(3, 8), C13);

    _mm256_storeu_ps(&Ctrow(0, 16), C20); 
    _mm256_storeu_ps(&Ctrow(1, 16), C21); 
    _mm256_storeu_ps(&Ctrow(2, 16), C22);
    _mm256_storeu_ps(&Ctrow(3, 16), C23);

    for ( j=0; j<nr; j++ )
      for ( i=0; i<mr; i++ )
        Crow(i,j) = Ctrow(i,j);
  
  } else {

    _mm256_storeu_ps(&Crow(0, 0), C00);
    _mm256_storeu_ps(&Crow(1, 0), C01);
    _mm256_storeu_ps(&Crow(2, 0), C02);
    _mm256_storeu_ps(&Crow(3, 0), C03);
    
    _mm256_storeu_ps(&Crow(0, 8), C10);
    _mm256_storeu_ps(&Crow(1, 8), C11);
    _mm256_storeu_ps(&Crow(2, 8), C12);
    _mm256_storeu_ps(&Crow(3, 8), C13);
     
    _mm256_storeu_ps(&Crow(0, 16), C20);
    _mm256_storeu_ps(&Crow(1, 16), C21);
    _mm256_storeu_ps(&Crow(2, 16), C22);
    _mm256_storeu_ps(&Crow(3, 16), C23);

  }
}

//-------------------------------------------------------------------------------------------------//
//*************************************************************************************************//
//-------------------------------------------------------------------------------------------------//

void gemm_microkernel_Cresident_AMD_avx256_32x2_fp32( char orderC, int mr, int nr, int kc, float alpha, 
		                                              float *Ar, float *Br, float beta, float *C, int ldC ) {

  int i, j, k, baseB = 0, baseA = 0, ldCt = MR, Amr, Bnr;
  __m256 C00, C01,
         C10, C11,
         C20, C21,
         C30, C31;
  
  __m256 A0, A1, A2, A3, B0;

  float   zero = 0.0, *Aptr, *Bptr, Ctmp[MR*NR];

  if ( kc==0 )
    return;

  if ( beta!=zero ) {
    C00  = _mm256_loadu_ps(&Ccol(0,0)); 
    C01  = _mm256_loadu_ps(&Ccol(0,1));
  
    C10  = _mm256_loadu_ps(&Ccol(8,0));
    C11  = _mm256_loadu_ps(&Ccol(8,1));
    
    C20  = _mm256_loadu_ps(&Ccol(16,0));
    C21  = _mm256_loadu_ps(&Ccol(16,1));
    
    C30  = _mm256_loadu_ps(&Ccol(24,0));
    C31  = _mm256_loadu_ps(&Ccol(24,1));
    
  } else {
    C00  = _mm256_set1_ps ( 0 ); 
    C01  = _mm256_set1_ps ( 0 ); 
  
    C10  = _mm256_set1_ps ( 0 ); 
    C11  = _mm256_set1_ps ( 0 ); 
    
    C20  = _mm256_set1_ps ( 0 ); 
    C21  = _mm256_set1_ps ( 0 ); 
    
    C30  = _mm256_set1_ps ( 0 ); 
    C31  = _mm256_set1_ps ( 0 ); 
  }

  Aptr = &Ar[0];
  Bptr = &Br[0];
  Amr  = MR;
  Bnr  = NR;

  if ( alpha!=zero ) {
    //if (mr == MR && nr == NR) 
      for ( k=0; k < kc; k++ ) {
        // Load colums/rows of A/B for current iteration
        A0 = _mm256_loadu_ps(&Aptr[baseA + 0]);
        A1 = _mm256_loadu_ps(&Aptr[baseA + 8]);
        A2 = _mm256_loadu_ps(&Aptr[baseA + 16]);
        A3 = _mm256_loadu_ps(&Aptr[baseA + 24]);

        B0 = _mm256_broadcast_ss(&Bptr[baseB + 0]);
        C00 = _mm256_fmadd_ps(A0, B0, C00);
        C10 = _mm256_fmadd_ps(A1, B0, C10);
        C20 = _mm256_fmadd_ps(A2, B0, C20);
        C30 = _mm256_fmadd_ps(A3, B0, C30);

        B0 = _mm256_broadcast_ss(&Bptr[baseB + 1]);
        C01 = _mm256_fmadd_ps(A0, B0, C01);
        C11 = _mm256_fmadd_ps(A1, B0, C11);
        C21 = _mm256_fmadd_ps(A2, B0, C21);
        C31 = _mm256_fmadd_ps(A3, B0, C31);
      
        baseA = baseA+Amr;
        baseB = baseB+Bnr;
      }
  }

  if ((mr < MR) || (nr < NR)) {
   
   _mm256_storeu_ps(&Ctref(0, 0), C00);
   _mm256_storeu_ps(&Ctref(0, 1), C01);

   _mm256_storeu_ps(&Ctref(8, 0), C10);
   _mm256_storeu_ps(&Ctref(8, 1), C11);

   _mm256_storeu_ps(&Ctref(16, 0), C20);
   _mm256_storeu_ps(&Ctref(16, 1), C21);
   
   _mm256_storeu_ps(&Ctref(24, 0), C30);
   _mm256_storeu_ps(&Ctref(24, 1), C31);
   
   for ( j=0; j<nr; j++ )
     for ( i=0; i<mr; i++ )
       Ccol(i,j) = Ctcol(i,j);
  
  } else {
    _mm256_storeu_ps(&Ccol(0,0), C00);
    _mm256_storeu_ps(&Ccol(0,1), C01);
    
    _mm256_storeu_ps(&Ccol(8,0), C10);
    _mm256_storeu_ps(&Ccol(8,1), C11);
    
    _mm256_storeu_ps(&Ccol(16,0), C20);
    _mm256_storeu_ps(&Ccol(16,1), C21);
    
    _mm256_storeu_ps(&Ccol(24,0), C30);
    _mm256_storeu_ps(&Ccol(24,1), C31);
  }
}

void gemm_microkernel_Cresident_AMD_avx256_40x2_fp32( char orderC, int mr, int nr, int kc, float alpha, 
		                                              float *Ar, float *Br, float beta, float *C, int ldC ) {

  int i, j, k, baseB = 0, baseA = 0, ldCt = MR, Amr, Bnr;
  __m256 C00, C01,
         C10, C11,
         C20, C21,
         C30, C31,
	 C40, C41;
  
  __m256 A0, A1, A2, B0, B1;

  float   zero = 0.0, *Aptr, *Bptr, Ctmp[MR*NR];

  if ( kc==0 )
    return;

  if ( beta!=zero ) {
    C00  = _mm256_loadu_ps(&Ccol(0,0)); 
    C01  = _mm256_loadu_ps(&Ccol(0,1));
  
    C10  = _mm256_loadu_ps(&Ccol(8,0));
    C11  = _mm256_loadu_ps(&Ccol(8,1));
    
    C20  = _mm256_loadu_ps(&Ccol(16,0));
    C21  = _mm256_loadu_ps(&Ccol(16,1));
    
    C30  = _mm256_loadu_ps(&Ccol(24,0));
    C31  = _mm256_loadu_ps(&Ccol(24,1));
    
    C40  = _mm256_loadu_ps(&Ccol(32,0));
    C41  = _mm256_loadu_ps(&Ccol(32,1));

  } else {
    C00  = _mm256_set1_ps ( 0 ); 
    C01  = _mm256_set1_ps ( 0 ); 
  
    C10  = _mm256_set1_ps ( 0 ); 
    C11  = _mm256_set1_ps ( 0 ); 
    
    C20  = _mm256_set1_ps ( 0 ); 
    C21  = _mm256_set1_ps ( 0 ); 
    
    C30  = _mm256_set1_ps ( 0 ); 
    C31  = _mm256_set1_ps ( 0 ); 
    
    C40  = _mm256_set1_ps ( 0 ); 
    C41  = _mm256_set1_ps ( 0 ); 
  }

  Aptr = &Ar[0];
  Bptr = &Br[0];
  Amr  = MR;
  Bnr  = NR;

  if ( alpha!=zero ) {
    //if (mr == MR && nr == NR) 
      for ( k=0; k < kc; k++ ) {
        // Load colums/rows of A/B for current iteration
        A0 = _mm256_loadu_ps(&Aptr[baseA + 0]);
        A1 = _mm256_loadu_ps(&Aptr[baseA + 8]);

        B0 = _mm256_broadcast_ss(&Bptr[baseB + 0]);
        B1 = _mm256_broadcast_ss(&Bptr[baseB + 1]);
        
	C00 = _mm256_fmadd_ps(A0, B0, C00);
        C10 = _mm256_fmadd_ps(A1, B0, C10);

	C01 = _mm256_fmadd_ps(A0, B1, C01);
        C11 = _mm256_fmadd_ps(A1, B1, C11);

        A0 = _mm256_loadu_ps(&Aptr[baseA + 16]);
        A1 = _mm256_loadu_ps(&Aptr[baseA + 24]);
        A2 = _mm256_loadu_ps(&Aptr[baseA + 32]);
        
        C20 = _mm256_fmadd_ps(A0, B0, C20);
        C30 = _mm256_fmadd_ps(A1, B0, C30);
        C40 = _mm256_fmadd_ps(A2, B0, C40);
        
	C21 = _mm256_fmadd_ps(A0, B1, C21);
        C31 = _mm256_fmadd_ps(A1, B1, C31);
        C41 = _mm256_fmadd_ps(A2, B1, C41);
      
        baseA = baseA+Amr;
        baseB = baseB+Bnr;
      }
  }

  if ((mr < MR) || (nr < NR)) {
   
   _mm256_storeu_ps(&Ctref(0, 0), C00);
   _mm256_storeu_ps(&Ctref(0, 1), C01);

   _mm256_storeu_ps(&Ctref(8, 0), C10);
   _mm256_storeu_ps(&Ctref(8, 1), C11);

   _mm256_storeu_ps(&Ctref(16, 0), C20);
   _mm256_storeu_ps(&Ctref(16, 1), C21);
   
   _mm256_storeu_ps(&Ctref(24, 0), C30);
   _mm256_storeu_ps(&Ctref(24, 1), C31);
   
   _mm256_storeu_ps(&Ctref(32, 0), C40);
   _mm256_storeu_ps(&Ctref(32, 1), C41);

   for ( j=0; j<nr; j++ )
     for ( i=0; i<mr; i++ )
       Ccol(i,j) = Ctcol(i,j);
  
  } else {
    _mm256_storeu_ps(&Ccol(0,0), C00);
    _mm256_storeu_ps(&Ccol(0,1), C01);
    
    _mm256_storeu_ps(&Ccol(8,0), C10);
    _mm256_storeu_ps(&Ccol(8,1), C11);
    
    _mm256_storeu_ps(&Ccol(16,0), C20);
    _mm256_storeu_ps(&Ccol(16,1), C21);
    
    _mm256_storeu_ps(&Ccol(24,0), C30);
    _mm256_storeu_ps(&Ccol(24,1), C31);
    
    _mm256_storeu_ps(&Ccol(32,0), C40);
    _mm256_storeu_ps(&Ccol(32,1), C41);
  }
}


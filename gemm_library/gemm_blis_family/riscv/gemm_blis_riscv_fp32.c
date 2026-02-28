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
#include <riscv_vector.h>

#include "blis.h"
#include "gemm_blis_riscv_fp32.h"

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



inline void gemm_microkernel_Cresident_riscv_4x4_prefetch_fp32( char orderC, int mr, int nr, int kc, float alpha, float *Ar, float *Br, float beta, float *C, int ldC ){
/*
  BLIS GEMM microkernel, computes the product Cr := Cr + Ar * Br
  Specific: only for MRxNR = 4x4
*/
  int         i, j, k, baseA = 0, baseB = 0, ldCt = MR, Amr, Bnr;
  int vlength = 4;
  float init[4] = {0.0,0.0,0.0,0.0};
  float * init_ptr = init;
  size_t vl =  vsetvl_e32m1(vlength); 
  /*float32x4_t C0, C1, C2, C3, 
	      A0, A1, A2, A3, A0n, B0, B0n;*/
  vfloat32m1_t C0= vle32_v_f32m1(init_ptr, vl); 
  vfloat32m1_t C1= vle32_v_f32m1(init_ptr, vl); 
  vfloat32m1_t C2= vle32_v_f32m1(init_ptr, vl); 
  vfloat32m1_t C3= vle32_v_f32m1(init_ptr, vl); 
  vfloat32m1_t A0; 
  vfloat32m1_t A1; 
  vfloat32m1_t A2; 
  vfloat32m1_t A3; 
  vfloat32m1_t A0n;
  vfloat32m1_t B0n; // = vundefined_f32m1();
  
  float b00, b01, b02, b03;
  
  float       zero = 0.0, one = 1.0, *Aptr, *Bptr, Ctmp[MR*NR];
  if ( kc==0 )
    return;
  /*
  NOT NEEDED IN RISCV.
  They are initializad at the declaration.
  C0 = vmovq_n_f32(0);
  C1 = vmovq_n_f32(0);
  C2 = vmovq_n_f32(0);
  C3 = vmovq_n_f32(0);
  */
  
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
  //A0 = //vld1q_f32(&Aptr[0]);
  A0 = vle32_v_f32m1(Aptr, vl); 
  B0n = vle32_v_f32m1(Bptr, vl); 
  vsse32_v_f32m1(&b00,0, B0n, 1);
  vsse32_v_f32m1(&b01,0, B0n, 2);
  vsse32_v_f32m1(&b02,0, B0n, 3);
  vsse32_v_f32m1(&b03,0, B0n, 4);
  
  // Iterate from 1 to kc-1 (last iteration outside loop)
  if ( alpha!=zero ) {
    for ( k=0; k<kc-1; k++ ) {
      // Prefect colum/row of A/B for next iteration
      baseA = baseA+Amr;
      baseB = baseB+Bnr;
      //A0n = vld1q_f32(&Aptr[baseA]);
      A0n = vle32_v_f32m1(&Aptr[baseA], vl); 
      B0n = vle32_v_f32m1(&Bptr[baseB], vl); 
      
       //C0 = vfmaq_laneq_f32(C0, A0, B0, 0);
      //C1 = vfmaq_laneq_f32(C1, A0, B0, 1);
      //C2 = vfmaq_laneq_f32(C2, A0, B0, 2);
      //C3 = vfmaq_laneq_f32(C3, A0, B0, 3);
      C0 = vfmacc_vf_f32m1(C0, b00, A0, vl);
      C1 = vfmacc_vf_f32m1(C1, b01, A0, vl);
      C2 = vfmacc_vf_f32m1(C2, b02, A0, vl);
      C3 = vfmacc_vf_f32m1(C3, b03, A0, vl);

      A0 = A0n;
      vsse32_v_f32m1(&b00,0, B0n, 1);
      vsse32_v_f32m1(&b01,0, B0n, 2);
      vsse32_v_f32m1(&b02,0, B0n, 3);
      vsse32_v_f32m1(&b03,0, B0n, 4);
    }
    // Last iteration
    //C0 = vfmaq_laneq_f32(C0, A0, B0, 0);
    //C1 = vfmaq_laneq_f32(C1, A0, B0, 1);
    //C2 = vfmaq_laneq_f32(C2, A0, B0, 2);
    //C3 = vfmaq_laneq_f32(C3, A0, B0, 3);
    C0 = vfmacc_vf_f32m1(C0, b00, A0, vl);
    C1 = vfmacc_vf_f32m1(C1, b01, A0, vl);
    C2 = vfmacc_vf_f32m1(C2, b02, A0, vl);
    C3 = vfmacc_vf_f32m1(C3, b03, A0, vl);
    
    if ( alpha==-one ) {
      // It works with riscv
      C0 = -C0; C1 = -C1; C2 = -C2; C3 = -C3; 
    }
    else if ( alpha!=one ) {
      // It works with riscv
      C0 = alpha*C0; C1 = alpha*C1; C2 = alpha*C2; C3 = alpha*C3;
    }
  }

  if ( (mr<MR)||(nr<NR) ) {
    //vst1q_f32(&Ctref(0,0), C0);
    //vst1q_f32(&Ctref(0,1), C1);
    //vst1q_f32(&Ctref(0,2), C2);
    //vst1q_f32(&Ctref(0,3), C3);
    vse32_v_f32m1(&Ctref(0,0), C0, vl);
    vse32_v_f32m1(&Ctref(0,1), C1, vl);
    vse32_v_f32m1(&Ctref(0,2), C2, vl);
    vse32_v_f32m1(&Ctref(0,3), C3, vl);

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
     
     // A0 = vld1q_f32(&Ccol(0,0));
     // A1 = vld1q_f32(&Ccol(0,1));
     // A2 = vld1q_f32(&Ccol(0,2));
     // A3 = vld1q_f32(&Ccol(0,3));
      A0 = vle32_v_f32m1(&Ccol(0,0), vl);
      A1 = vle32_v_f32m1(&Ccol(0,1), vl);
      A2 = vle32_v_f32m1(&Ccol(0,2), vl);
      A3 = vle32_v_f32m1(&Ccol(0,3), vl);

     // C0 = beta*A0 + C0;
     // C1 = beta*A1 + C1;
     // C2 = beta*A2 + C2;
     // C3 = beta*A3 + C3;

     C0= vfmacc_vf_f32m1(C0, beta, A0, vl);
     C1= vfmacc_vf_f32m1(C1, beta, A1, vl);
     C2= vfmacc_vf_f32m1(C2, beta, A2, vl);
     C3= vfmacc_vf_f32m1(C3, beta, A3, vl);

    }
    //vst1q_f32(&Ccol(0,0), C0);
    //vst1q_f32(&Ccol(0,1), C1);
    //vst1q_f32(&Ccol(0,2), C2);
    //vst1q_f32(&Ccol(0,3), C3);
    vse32_v_f32m1(&Ccol(0,0), C0, vl);
    vse32_v_f32m1(&Ccol(0,1), C1, vl);
    vse32_v_f32m1(&Ccol(0,2), C2, vl);
    vse32_v_f32m1(&Ccol(0,3), C3, vl);
  }
  else {
    printf("Error: Incorrect use of 4x4 micro-kernel with %d x %d block\n", mr, nr);
    exit(-1);
  }
}

inline void gemm_microkernel_Cresident_riscv_4x4_prefetch_unroll_fp32( char orderC, int mr, int nr, int kc, float alpha, float *Ar, float *Br, float beta, float *C, int ldC ){
/////
//  BLIS GEMM microkernel, computes the product Cr := Cr + Ar * Br
//  Specific: only for MRxNR = 4x4
////
  int         i, j, k, baseA = 0, baseB = 0, ldCt = MR, Amr, Bnr;
  int vlength = 4;
  float init[4] = {0.0,0.0,0.0,0.0};
  float * init_ptr = init;
  size_t vl =  vsetvl_e32m1(vlength); 

  vfloat32m1_t C0= vle32_v_f32m1(init_ptr, vl); 
  vfloat32m1_t C1= vle32_v_f32m1(init_ptr, vl); 
  vfloat32m1_t C2= vle32_v_f32m1(init_ptr, vl); 
  vfloat32m1_t C3= vle32_v_f32m1(init_ptr, vl); 
  vfloat32m1_t A0; 
  vfloat32m1_t A1; 
  vfloat32m1_t A2; 
  vfloat32m1_t A3; 
  vfloat32m1_t A0n;
  vfloat32m1_t B0n; // = vundefined_f32m1();
  
  float b00, b01, b02, b03;
  float       zero = 0.0, one = 1.0, *Aptr, *Bptr, Ctmp[MR*NR];

  if ( kc==0 )
    return;
  
 
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
  A0  = vle32_v_f32m1(Aptr, vl); 
  B0n = vle32_v_f32m1(Bptr,vl);

  vsse32_v_f32m1(&b00,0, B0n, 1); //load lane[0]----> b00
  vsse32_v_f32m1(&b01,0, B0n, 2);
  vsse32_v_f32m1(&b02,0, B0n, 3);
  vsse32_v_f32m1(&b03,0, B0n, 4);

  if ( alpha!=zero ) {
    for ( k=0; k<kc-2; k+=2 ) {

      baseA = baseA+Amr;
      baseB = baseB+Bnr;

      A0n = vle32_v_f32m1(&Aptr[baseA],vl);
      B0n = vle32_v_f32m1(&Bptr[baseB],vl);
       
      C0 = vfmacc_vf_f32m1(C0, b00, A0, vl);
      C1 = vfmacc_vf_f32m1(C1, b01, A0, vl);
      C2 = vfmacc_vf_f32m1(C2, b02, A0, vl);
      C3 = vfmacc_vf_f32m1(C3, b03, A0, vl);
    
      A0 = A0n;
      vsse32_v_f32m1(&b00,0, B0n, 1);
      vsse32_v_f32m1(&b01,0, B0n, 2);
      vsse32_v_f32m1(&b02,0, B0n, 3);
      vsse32_v_f32m1(&b03,0, B0n, 4);
      
      baseA = baseA+Amr;
      baseB = baseB+Bnr;

      A0n = vle32_v_f32m1(&Aptr[baseA],vl);
      B0n = vle32_v_f32m1(&Bptr[baseB],vl);
       
      C0 = vfmacc_vf_f32m1(C0, b00, A0, vl);
      C1 = vfmacc_vf_f32m1(C1, b01, A0, vl);
      C2 = vfmacc_vf_f32m1(C2, b02, A0, vl);
      C3 = vfmacc_vf_f32m1(C3, b03, A0, vl);

      A0 = A0n;
      vsse32_v_f32m1(&b00,0, B0n, 1);
      vsse32_v_f32m1(&b01,0, B0n, 2);
      vsse32_v_f32m1(&b02,0, B0n, 3);
      vsse32_v_f32m1(&b03,0, B0n, 4);
    }
  
    if (kc % 2 == 0) {
      baseA = baseA+Amr;
      baseB = baseB+Bnr;

      A0n = vle32_v_f32m1(&Aptr[baseA], vl); 
      B0n = vle32_v_f32m1(&Bptr[baseB], vl);  
    }
    	
      C0 = vfmacc_vf_f32m1(C0, b00, A0, vl);
      C1 = vfmacc_vf_f32m1(C1, b01, A0, vl);
      C2 = vfmacc_vf_f32m1(C2, b02, A0, vl);
      C3 = vfmacc_vf_f32m1(C3, b03, A0, vl);
	
    if (kc % 2 == 0) {
      
      A0 = A0n;
      vsse32_v_f32m1(&b00,0, B0n, 1);
      vsse32_v_f32m1(&b01,0, B0n, 2);
      vsse32_v_f32m1(&b02,0, B0n, 3);
      vsse32_v_f32m1(&b03,0, B0n, 4);

      C0 = vfmacc_vf_f32m1(C0, b00, A0, vl);
      C1 = vfmacc_vf_f32m1(C1, b01, A0, vl);
      C2 = vfmacc_vf_f32m1(C2, b02, A0, vl);
      C3 = vfmacc_vf_f32m1(C3, b03, A0, vl);
    }
  
    if ( alpha==-one ) {
      C0 = -C0; C1 = -C1; C2 = -C2; C3 = -C3; 
    }
    else if ( alpha!=one ) {
      C0 = alpha*C0; C1 = alpha*C1; C2 = alpha*C2; C3 = alpha*C3;
    }
  }

  if ( (mr<MR)||(nr<NR) ) {
   
   vse32_v_f32m1(&Ctref(0,0), C0, vl);
   vse32_v_f32m1(&Ctref(0,1), C1, vl);
   vse32_v_f32m1(&Ctref(0,2), C2, vl);
   vse32_v_f32m1(&Ctref(0,3), C3, vl);

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
	    
      A0 = vle32_v_f32m1(&Ccol(0,0), vl);
      A1 = vle32_v_f32m1(&Ccol(0,1), vl);
      A2 = vle32_v_f32m1(&Ccol(0,2), vl);
      A3 = vle32_v_f32m1(&Ccol(0,3), vl);
      
      C0= vfmacc_vf_f32m1(C0, beta, A0, vl);
      C1= vfmacc_vf_f32m1(C1, beta, A1, vl);
      C2= vfmacc_vf_f32m1(C2, beta, A2, vl);
      C3= vfmacc_vf_f32m1(C3, beta, A3, vl);
    }

    vse32_v_f32m1(&Ccol(0,0), C0, vl);
    vse32_v_f32m1(&Ccol(0,1), C1, vl);
    vse32_v_f32m1(&Ccol(0,2), C2, vl);
    vse32_v_f32m1(&Ccol(0,3), C3, vl);
  }
  else {
    printf("Error: Incorrect use of 4x4 micro-kernel with %d x %d block\n", mr, nr);
    exit(-1);
  }
}


inline void gemm_microkernel_Cresident_riscv_8x8_prefetch_fp32( char orderC, int mr, int nr, int kc, float alpha, float *Ar, float *Br, float beta, float *C, int ldC ){
////
//  BLIS GEMM microkernel, computes the product Cr := Cr + Ar * Br
//  Specific: only for MRxNR = 8x8
////
  int         i, j, k, baseA = 0, baseB = 0, ldCt = MR, Amr, Bnr;
  int vlength = 8;					
  float init[8] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
  float * init_ptr = init;
  size_t vl =  vsetvl_e32m2(vlength); 
  
  vfloat32m2_t C0= vle32_v_f32m2(init_ptr, vl); 	// C register 
  vfloat32m2_t C1= vle32_v_f32m2(init_ptr, vl); 
  vfloat32m2_t C2= vle32_v_f32m2(init_ptr, vl); 
  vfloat32m2_t C3= vle32_v_f32m2(init_ptr, vl); 
  vfloat32m2_t C4= vle32_v_f32m2(init_ptr, vl); 
  vfloat32m2_t C5= vle32_v_f32m2(init_ptr, vl); 
  vfloat32m2_t C6= vle32_v_f32m2(init_ptr, vl); 
  vfloat32m2_t C7= vle32_v_f32m2(init_ptr, vl); 
 
  vfloat32m2_t A0; 	// register A
  vfloat32m2_t A1; 
  vfloat32m2_t A2; 
  vfloat32m2_t A3; 
  vfloat32m2_t A4; 
  vfloat32m2_t A5; 
  vfloat32m2_t A6; 
  vfloat32m2_t A7;
 
  vfloat32m2_t A0n;	// Temporal register prefecth A
  vfloat32m2_t B0n; 	// Temporal register prefecth B

  float  b00, b01, b02, b03,	
	 b04, b05, b06, b07;	// var to index B index
  float  zero = 0.0, one = 1.0, *Aptr, *Bptr, Ctmp[MR*NR];

{
  float  zero = 0.0, one = 1.0, *Aptr, *Bptr, Ctmp[MR*NR];

  if ( kc==0 )
    return;

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
  A0  = vle32_v_f32m2(Aptr, vl); 
  B0n = vle32_v_f32m2(Bptr, vl); 
  vsse32_v_f32m2(&b00,0, B0n, 1);

  A0   = vle32_v_f32m2(Aptr, vl); 
  B0n  = vle32_v_f32m2(Bptr, vl);
 
  vsse32_v_f32m2(&b00,0, B0n, 1); // index data : B0n[n]--->b00
  vsse32_v_f32m2(&b01,0, B0n, 2);
  vsse32_v_f32m2(&b02,0, B0n, 3);
  vsse32_v_f32m2(&b03,0, B0n, 4);
  vsse32_v_f32m2(&b04,0, B0n, 5);
  vsse32_v_f32m2(&b05,0, B0n, 6);
  vsse32_v_f32m2(&b06,0, B0n, 7);
  vsse32_v_f32m2(&b07,0, B0n, 8);

  if ( alpha!=zero ) {
    for ( k=0; k<kc-1; k++ ) {

      baseA = baseA+Amr;
      baseB = baseB+Bnr;
     
      baseA +=Amr;
      baseB +=Bnr;
     
      A0n = vle32_v_f32m2(&Aptr[baseA], vl); 
      B0n = vle32_v_f32m2(&Bptr[baseB], vl); 
      
      C0 = vfmacc_vf_f32m2(C0, b00, A0, vl);
      C1 = vfmacc_vf_f32m2(C1, b01, A0, vl);
      C2 = vfmacc_vf_f32m2(C2, b02, A0, vl);
      C3 = vfmacc_vf_f32m2(C3, b03, A0, vl);
      C4 = vfmacc_vf_f32m2(C4, b04, A0, vl);
      C5 = vfmacc_vf_f32m2(C5, b05, A0, vl);
      C6 = vfmacc_vf_f32m2(C6, b06, A0, vl);
      C7 = vfmacc_vf_f32m2(C7, b07, A0, vl);

      A0 = A0n;// swap data temp ---> register
      vsse32_v_f32m2(&b00,0, B0n, 1);
      vsse32_v_f32m2(&b01,0, B0n, 2);
      vsse32_v_f32m2(&b02,0, B0n, 3);
      vsse32_v_f32m2(&b03,0, B0n, 4);
      vsse32_v_f32m2(&b04,0, B0n, 5);
      vsse32_v_f32m2(&b05,0, B0n, 6);
      vsse32_v_f32m2(&b06,0, B0n, 7);
      vsse32_v_f32m2(&b07,0, B0n, 8);
    }
  
    
    C0 = vfmacc_vf_f32m2(C0, b00, A0, vl);
    C1 = vfmacc_vf_f32m2(C1, b01, A0, vl);
    C2 = vfmacc_vf_f32m2(C2, b02, A0, vl);
    C3 = vfmacc_vf_f32m2(C3, b03, A0, vl);
    C4 = vfmacc_vf_f32m2(C4, b04, A0, vl);
    C5 = vfmacc_vf_f32m2(C5, b05, A0, vl);
    C6 = vfmacc_vf_f32m2(C6, b06, A0, vl);
    C7 = vfmacc_vf_f32m2(C7, b07, A0, vl);

    if ( alpha==-one ) {
      C0 = -C0; C1 = -C1; C2 = -C2; C3 = -C3; C4 = -C4; C5 = -C5; C6 = -C6; C7 = -C7; 
    }
    else if ( alpha!=one ) {
      C0 = alpha*C0; C1 = alpha*C1; C2 = alpha*C2; C3 = alpha*C3; C4 = alpha*C4; C5 = alpha*C5; C6 = alpha*C6; C7 = alpha*C7;
    }
  }

  if ( (mr<MR)||(nr<NR) ) {
    
    vse32_v_f32m2(&Ctref(0,0), C0, vl);
    vse32_v_f32m2(&Ctref(0,1), C1, vl);
    vse32_v_f32m2(&Ctref(0,2), C2, vl);
    vse32_v_f32m2(&Ctref(0,3), C3, vl);
    vse32_v_f32m2(&Ctref(0,4), C4, vl);
    vse32_v_f32m2(&Ctref(0,5), C5, vl);
    vse32_v_f32m2(&Ctref(0,6), C6, vl);
    vse32_v_f32m2(&Ctref(0,7), C7, vl);

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
      
      A0 = vle32_v_f32m2(&Ccol(0,0), vl);
      A1 = vle32_v_f32m2(&Ccol(0,1), vl);
      A2 = vle32_v_f32m2(&Ccol(0,2), vl);
      A3 = vle32_v_f32m2(&Ccol(0,3), vl);
      A4 = vle32_v_f32m2(&Ccol(0,4), vl);
      A5 = vle32_v_f32m2(&Ccol(0,5), vl);
      A6 = vle32_v_f32m2(&Ccol(0,6), vl);
      A7 = vle32_v_f32m2(&Ccol(0,7), vl);
     
     C0= vfmacc_vf_f32m2(C0, beta, A0, vl);
     C1= vfmacc_vf_f32m2(C1, beta, A1, vl);
     C2= vfmacc_vf_f32m2(C2, beta, A2, vl);
     C3= vfmacc_vf_f32m2(C3, beta, A3, vl);
     C4= vfmacc_vf_f32m2(C4, beta, A4, vl);
     C5= vfmacc_vf_f32m2(C5, beta, A5, vl);
     C6= vfmacc_vf_f32m2(C6, beta, A6, vl);
     C7= vfmacc_vf_f32m2(C7, beta, A7, vl);
    }
    
    vse32_v_f32m2(&Ccol(0,0), C0, vl);
    vse32_v_f32m2(&Ccol(0,1), C1, vl);
    vse32_v_f32m2(&Ccol(0,2), C2, vl);
    vse32_v_f32m2(&Ccol(0,3), C3, vl);
    vse32_v_f32m2(&Ccol(0,4), C4, vl);
    vse32_v_f32m2(&Ccol(0,5), C5, vl);
    vse32_v_f32m2(&Ccol(0,6), C6, vl);
    vse32_v_f32m2(&Ccol(0,7), C7, vl);
  }
  else {
    printf("Error: Incorrect use of 8x8 micro-kernel with %d x %d block\n", mr, nr);
    exit(-1);
  }
}
}

inline void gemm_microkernel_Cresident_riscv_8x8_prefetch_unroll_fp32( char orderC, int mr, int nr, int kc, float alpha, float *Ar, float *Br, float beta, float *C, int ldC ){
////
//  BLIS GEMM microkernel, computes the product Cr := Cr + Ar * Br
//  Specific: only for MRxNR = 8x8
////

  int         i, j, k, baseA = 0, baseB = 0, ldCt = MR, Amr, Bnr;
  int vlength = 8;					
  float init[8] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
  float * init_ptr = init;
  size_t vl =  vsetvl_e32m2(vlength); 
  
  vfloat32m2_t C0= vle32_v_f32m2(init_ptr, vl); 	// C register 
  vfloat32m2_t C1= vle32_v_f32m2(init_ptr, vl); 
  vfloat32m2_t C2= vle32_v_f32m2(init_ptr, vl); 
  vfloat32m2_t C3= vle32_v_f32m2(init_ptr, vl); 
  vfloat32m2_t C4= vle32_v_f32m2(init_ptr, vl); 
  vfloat32m2_t C5= vle32_v_f32m2(init_ptr, vl); 
  vfloat32m2_t C6= vle32_v_f32m2(init_ptr, vl); 
  vfloat32m2_t C7= vle32_v_f32m2(init_ptr, vl); 
  
  vfloat32m2_t A0; 	// register A
  vfloat32m2_t A1; 
  vfloat32m2_t A2; 
  vfloat32m2_t A3; 
  vfloat32m2_t A4; 
  vfloat32m2_t A5; 
  vfloat32m2_t A6; 
  vfloat32m2_t A7;
 
  vfloat32m2_t A0n;	// Temporal register prefecth A
  vfloat32m2_t B0n; 	// Temporal register prefecth B

  float  b00, b01, b02, b03,	
	 b04, b05, b06, b07;	// var to index B index
  float  zero = 0.0, one = 1.0, *Aptr, *Bptr, Ctmp[MR*NR];

{

  if ( kc==0 )
    return;

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

  A0   = vle32_v_f32m2(Aptr, vl); 
  B0n  = vle32_v_f32m2(Bptr, vl);
 
  vsse32_v_f32m2(&b00,0, B0n, 1); // index data : B0n[n]--->b00
  vsse32_v_f32m2(&b01,0, B0n, 2);
  vsse32_v_f32m2(&b02,0, B0n, 3);
  vsse32_v_f32m2(&b03,0, B0n, 4);
  vsse32_v_f32m2(&b04,0, B0n, 5);
  vsse32_v_f32m2(&b05,0, B0n, 6);
  vsse32_v_f32m2(&b06,0, B0n, 7);
  vsse32_v_f32m2(&b07,0, B0n, 8);

  if ( alpha!=zero ) {
    for ( k=0; k<kc-2; k+=2 ) {

      baseA +=Amr;
      baseB +=Bnr;
     
      A0n = vle32_v_f32m2(&Aptr[baseA], vl); 
      B0n = vle32_v_f32m2(&Bptr[baseB], vl); 
      
      C0 = vfmacc_vf_f32m2(C0, b00, A0, vl);
      C1 = vfmacc_vf_f32m2(C1, b01, A0, vl);
      C2 = vfmacc_vf_f32m2(C2, b02, A0, vl);
      C3 = vfmacc_vf_f32m2(C3, b03, A0, vl);
      C4 = vfmacc_vf_f32m2(C4, b04, A0, vl);
      C5 = vfmacc_vf_f32m2(C5, b05, A0, vl);
      C6 = vfmacc_vf_f32m2(C6, b06, A0, vl);
      C7 = vfmacc_vf_f32m2(C7, b07, A0, vl);

      A0 = A0n;				// swap data temp ---> register
      vsse32_v_f32m2(&b00,0, B0n, 1);	// swap and index register
      vsse32_v_f32m2(&b01,0, B0n, 2);
      vsse32_v_f32m2(&b02,0, B0n, 3);
      vsse32_v_f32m2(&b03,0, B0n, 4);
      vsse32_v_f32m2(&b04,0, B0n, 5);
      vsse32_v_f32m2(&b05,0, B0n, 6);
      vsse32_v_f32m2(&b06,0, B0n, 7);
      vsse32_v_f32m2(&b07,0, B0n, 8);
      
      baseA +=Amr;
      baseB +=Bnr;
     
      A0n = vle32_v_f32m2(&Aptr[baseA], vl); 
      B0n = vle32_v_f32m2(&Bptr[baseB], vl); 
      
      C0 = vfmacc_vf_f32m2(C0, b00, A0, vl);
      C1 = vfmacc_vf_f32m2(C1, b01, A0, vl);
      C2 = vfmacc_vf_f32m2(C2, b02, A0, vl);
      C3 = vfmacc_vf_f32m2(C3, b03, A0, vl);
      C4 = vfmacc_vf_f32m2(C4, b04, A0, vl);
      C5 = vfmacc_vf_f32m2(C5, b05, A0, vl);
      C6 = vfmacc_vf_f32m2(C6, b06, A0, vl);
      C7 = vfmacc_vf_f32m2(C7, b07, A0, vl);

      A0 = A0n;			 	// swap data temp ---> register
      vsse32_v_f32m2(&b00,0, B0n, 1);   // swap and index register
      vsse32_v_f32m2(&b01,0, B0n, 2);
      vsse32_v_f32m2(&b02,0, B0n, 3);
      vsse32_v_f32m2(&b03,0, B0n, 4);
      vsse32_v_f32m2(&b04,0, B0n, 5);
      vsse32_v_f32m2(&b05,0, B0n, 6);
      vsse32_v_f32m2(&b06,0, B0n, 7);
      vsse32_v_f32m2(&b07,0, B0n, 8);
    }
    
    if(kc % 2 == 0){
     baseA += Amr;
     baseB += Bnr;
     
     A0n = vle32_v_f32m2(&Aptr[baseA], vl);   
     B0n = vle32_v_f32m2(&Bptr[baseB], vl);    
    }

    C0 = vfmacc_vf_f32m2(C0, b00, A0, vl);
    C1 = vfmacc_vf_f32m2(C1, b01, A0, vl);
    C2 = vfmacc_vf_f32m2(C2, b02, A0, vl);
    C3 = vfmacc_vf_f32m2(C3, b03, A0, vl);
    C4 = vfmacc_vf_f32m2(C4, b04, A0, vl);
    C5 = vfmacc_vf_f32m2(C5, b05, A0, vl);
    C6 = vfmacc_vf_f32m2(C6, b06, A0, vl);
    C7 = vfmacc_vf_f32m2(C7, b07, A0, vl);
    
    if(kc % 2 == 0){
    
      A0 = A0n;			 	// swap data temp ---> register
      vsse32_v_f32m2(&b00,0, B0n, 1);   // swap and index register
      vsse32_v_f32m2(&b01,0, B0n, 2);
      vsse32_v_f32m2(&b02,0, B0n, 3);
      vsse32_v_f32m2(&b03,0, B0n, 4);
      vsse32_v_f32m2(&b04,0, B0n, 5);
      vsse32_v_f32m2(&b05,0, B0n, 6);
      vsse32_v_f32m2(&b06,0, B0n, 7);
      vsse32_v_f32m2(&b07,0, B0n, 8);

      C0 = vfmacc_vf_f32m2(C0, b00, A0, vl);
      C1 = vfmacc_vf_f32m2(C1, b01, A0, vl);
      C2 = vfmacc_vf_f32m2(C2, b02, A0, vl);
      C3 = vfmacc_vf_f32m2(C3, b03, A0, vl);
      C4 = vfmacc_vf_f32m2(C4, b04, A0, vl);
      C5 = vfmacc_vf_f32m2(C5, b05, A0, vl);
      C6 = vfmacc_vf_f32m2(C6, b06, A0, vl);
      C7 = vfmacc_vf_f32m2(C7, b07, A0, vl);
    } 

    if ( alpha==-one ) {
      C0 = -C0; C1 = -C1; C2 = -C2; C3 = -C3; C4 = -C4; C5 = -C5; C6 = -C6; C7 = -C7; 
    }
    else if ( alpha!=one ) {
      C0 = alpha*C0; C1 = alpha*C1; C2 = alpha*C2; C3 = alpha*C3; C4 = alpha*C4; C5 = alpha*C5; C6 = alpha*C6; C7 = alpha*C7;
    }
  }

  if ( (mr<MR)||(nr<NR) ) {
    
    vse32_v_f32m2(&Ctref(0,0), C0, vl);
    vse32_v_f32m2(&Ctref(0,1), C1, vl);
    vse32_v_f32m2(&Ctref(0,2), C2, vl);
    vse32_v_f32m2(&Ctref(0,3), C3, vl);
    vse32_v_f32m2(&Ctref(0,4), C4, vl);
    vse32_v_f32m2(&Ctref(0,5), C5, vl);
    vse32_v_f32m2(&Ctref(0,6), C6, vl);
    vse32_v_f32m2(&Ctref(0,7), C7, vl);

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
          
    	A0 = vle32_v_f32m2(&Ccol(0,0), vl);
      	A1 = vle32_v_f32m2(&Ccol(0,1), vl);
      	A2 = vle32_v_f32m2(&Ccol(0,2), vl);
      	A3 = vle32_v_f32m2(&Ccol(0,3), vl);
      	A4 = vle32_v_f32m2(&Ccol(0,4), vl);
      	A5 = vle32_v_f32m2(&Ccol(0,5), vl);
      	A6 = vle32_v_f32m2(&Ccol(0,6), vl);
      	A7 = vle32_v_f32m2(&Ccol(0,7), vl);
     
     	C0= vfmacc_vf_f32m2(C0, beta, A0, vl);
     	C1= vfmacc_vf_f32m2(C1, beta, A1, vl);
     	C2= vfmacc_vf_f32m2(C2, beta, A2, vl);
     	C3= vfmacc_vf_f32m2(C3, beta, A3, vl);
     	C4= vfmacc_vf_f32m2(C4, beta, A4, vl);
     	C5= vfmacc_vf_f32m2(C5, beta, A5, vl);
     	C6= vfmacc_vf_f32m2(C6, beta, A6, vl);
     	C7= vfmacc_vf_f32m2(C7, beta, A7, vl);
    }
    
    vse32_v_f32m2(&Ccol(0,0), C0, vl);
    vse32_v_f32m2(&Ccol(0,1), C1, vl);
    vse32_v_f32m2(&Ccol(0,2), C2, vl);
    vse32_v_f32m2(&Ccol(0,3), C3, vl);
    vse32_v_f32m2(&Ccol(0,4), C4, vl);
    vse32_v_f32m2(&Ccol(0,5), C5, vl);
    vse32_v_f32m2(&Ccol(0,6), C6, vl);
    vse32_v_f32m2(&Ccol(0,7), C7, vl);
  }
  else {
    printf("Error: Incorrect use of 8x8 micro-kernel with %d x %d block\n", mr, nr);
    exit(-1);
  }
}
}

/*
inline void gemm_microkernel_ABresident_riscv_4x4_fp32( char orderA, char transA, int mr, int nc, int kr, float alpha, float *A, int ldA, float *Br, float beta, float *Cr ){
//////
//  BLIS GEMM microkernel, computes the product Cr := beta * Cr + alpha * Ar * Br
//  Specific: only for MRxKR = 4x4
//  Cr --> mr x nc
//  Ar --> mr x kr
//  Br --> kr x nc
/////  

  int         i, j, k, baseB, baseC, Atlda = MR;
  float32x4_t A0, A1, A2, A3,
	      C0, B0;
  float       zero = 0.0, one = 1.0, Atmp[MR*KR];

  if ( nc==0 )
    return;

  if ( alpha==zero ) {
    if ( mr==MR ) {
      for ( j=0; j<nc; j++ ) {
        baseC = j*MR;
        C0 = vld1q_f32(&Cr[baseC]);
        if ( beta==-one )
          C0 = -C0; 
        else if ( beta!=one )
          C0 = beta*C0; 
        vst1q_f32(&Cr[baseC], C0);
      }
    }
    else {
      for ( j=0; j<nc; j++ ) {
        baseC = j*MR;
        for ( i=0; i<mr; i++ )
          if ( beta==-one )
            Cr[baseC+i] = -Cr[baseC+i];
          else if ( beta!=one )
            Cr[baseC+i] = beta*Cr[baseC+i];
      }
    }
  }
  else {
    if ( (mr<MR)||(kr<KR) ) {
      if ( transA=='N' ) {
        if ( orderA=='C' ) {
          for ( k=0; k<kr; k++ ) {
            for ( i=0; i<mr; i++ ) 
              Atref(i,k) = Acol(i,k);
            for ( i=mr; i<MR; i++ ) 
              Atref(i,k) = zero;
          }
          for ( k=kr; k<NR; k++ )
            for ( i=0; i<MR; i++ ) 
              Atref(i,k) = zero;
        }
        else {
          for ( k=0; k<kr; k++ ) {
            for ( i=0; i<mr; i++ ) 
              Atref(i,k) = Arow(i,k);
            for ( i=mr; i<MR; i++ ) 
              Atref(i,k) = zero;
          }
          for ( k=kr; k<NR; k++ )
            for ( i=0; i<MR; i++ ) 
              Atref(i,k) = zero;
        }
      }
      else {
        if ( orderA=='C' ) {
          for ( k=0; k<kr; k++ ) {
            for ( i=0; i<mr; i++ ) 
              Atref(i,k) = Acol(k,i);
            for ( i=mr; i<MR; i++ ) 
              Atref(i,k) = zero;
          }
          for ( k=kr; k<NR; k++ )
            for ( i=0; i<MR; i++ ) 
              Atref(i,k) = zero;
        }
        else {
          for ( k=0; k<kr; k++ ) {
            for ( i=0; i<mr; i++ ) 
              Atref(i,k) = Arow(k,i);
            for ( i=mr; i<MR; i++ ) 
              Atref(i,k) = zero;
          }
          for ( k=kr; k<NR; k++ )
            for ( i=0; i<MR; i++ ) 
              Atref(i,k) = zero;
        }
      }

      A0 = vld1q_f32(&Atref(0,0));
      A1 = vld1q_f32(&Atref(0,1));
      A2 = vld1q_f32(&Atref(0,2));
      A3 = vld1q_f32(&Atref(0,3));
    }
    else if ( (mr==MR)&&(kr==KR) ) {
      A0 = vld1q_f32(&Acol(0,0));
      A1 = vld1q_f32(&Acol(0,1));
      A2 = vld1q_f32(&Acol(0,2));
      A3 = vld1q_f32(&Acol(0,3));
      if ( ((transA=='T')&&(orderA=='C'))||
           ((transA=='N')&&(orderA=='R')) )
        fvtrans_float32_4x4( &A0, &A1, &A2, &A3 );
    }
    else {
      printf("Error: Incorrect use of 4x4 micro-kernel with %d x %d block\n", mr, kr);
      exit(-1);
    }

    if ( alpha==-one ) {
      A0 = -A0; A1 = -A1; A2 = -A2; A3 = -A3; 
    }
    else if ( alpha!=one ) {
      A0 = alpha*A0; A1 = alpha*A1; A2 = alpha*A2; A3 = alpha*A3; 
    }

    if ( beta==zero ) {
      // Iterate from 1 to nc
      for ( j=0; j<nc; j++ ) {

        baseC = j*MR;
        baseB = j*KR;
        C0 = vmovq_n_f32(0);
        B0 = vld1q_f32(&Br[baseB]);
     
        C0 = vfmaq_laneq_f32(C0, A0, B0, 0);
        C0 = vfmaq_laneq_f32(C0, A1, B0, 1);
        C0 = vfmaq_laneq_f32(C0, A2, B0, 2);
        C0 = vfmaq_laneq_f32(C0, A3, B0, 3);

        vst1q_f32(&Cr[baseC], C0);
      }
    }
    else {
      // Iterate from 1 to nc
      for ( j=0; j<nc; j++ ) {

        baseC = j*MR;
        baseB = j*KR;
        C0 = vld1q_f32(&Cr[baseC]);
        C0 = beta*C0;
        B0 = vld1q_f32(&Br[baseB]);
     
        C0 = vfmaq_laneq_f32(C0, A0, B0, 0);
        C0 = vfmaq_laneq_f32(C0, A1, B0, 1);
        C0 = vfmaq_laneq_f32(C0, A2, B0, 2);
        C0 = vfmaq_laneq_f32(C0, A3, B0, 3);

        vst1q_f32(&Cr[baseC], C0);
      } 
    }
  }
}

inline void fvtrans_float32_4x4( float32x4_t *A0, float32x4_t *A1, float32x4_t *A2, float32x4_t *A3 ) {
  float32x4x2_t V = vtrnq_f32 ( (float32x4_t) vtrn1q_f64 ( (float64x2_t) *A0, (float64x2_t) *A2 ),
                                (float32x4_t) vtrn1q_f64 ( (float64x2_t) *A1, (float64x2_t) *A3 ));
  float32x4x2_t W = vtrnq_f32 ( (float32x4_t) vtrn2q_f64 ( (float64x2_t) *A0, (float64x2_t) *A2 ),
                                (float32x4_t) vtrn2q_f64 ( (float64x2_t) *A1, (float64x2_t) *A3 ));
  *A0 = V.val[0];
  *A1 = V.val[1];
  *A2 = W.val[0];
  *A3 = W.val[1];
}

inline void fvtrans_float32_8x8( float32x4_t *A00, float32x4_t *A01, float32x4_t *A02, float32x4_t *A03,
                                 float32x4_t *A04, float32x4_t *A05, float32x4_t *A06, float32x4_t *A07,
                                 float32x4_t *A10, float32x4_t *A11, float32x4_t *A12, float32x4_t *A13,
                                 float32x4_t *A14, float32x4_t *A15, float32x4_t *A16, float32x4_t *A17 ) {
  fvtrans_float32_4x4( A00, A01, A02, A03 );
  fvtrans_float32_4x4( A10, A11, A12, A13 );
  fvtrans_float32_4x4( A04, A05, A06, A07);
  fvtrans_float32_4x4( A14, A15, A16, A17);
}
*/

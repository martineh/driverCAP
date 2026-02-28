#include "ukernels_sve.h"

//======================================================================================
//**************************************************************************************
//*                      M I C R O - K E R N E L    32 X 10                            *
//**************************************************************************************
//======================================================================================

#define GEMM_FMLA_LD1R_32X10(C00, C01, C02, C03, C04, C05, C06, C07, C08, C09, \
			     C10, C11, C12, C13, C14, C15, C16, C17, C18, C19, \
			     B0,   B1,  B2,  B3,  B4,			\
			     A0, A1, PZ, B, BASE, DESP)			\
  {            				  \
      C00 = svmla_f32_z(PZ, C00, A0, B0); \
      C10 = svmla_f32_z(PZ, C10, A1, B0); \
      B0  = svdup_n_f32(B[BASE + 5]);	  \
      					  \
      C01 = svmla_f32_z(PZ, C01, A0, B1); \
      C11 = svmla_f32_z(PZ, C11, A1, B1); \
      B1  = svdup_n_f32(B[BASE + 6]);	  \      					  
					  \
      C02 = svmla_f32_z(PZ, C02, A0, B2); \
      C12 = svmla_f32_z(PZ, C12, A1, B2); \
      B2  = svdup_n_f32(B[BASE + 7]);	  \
					  \
      C03 = svmla_f32_z(PZ, C03, A0, B3); \
      C13 = svmla_f32_z(PZ, C13, A1, B3); \
      B3  = svdup_n_f32(B[BASE + 8]);	  \
					  \
      C04 = svmla_f32_z(PZ, C04, A0, B4); \
      C14 = svmla_f32_z(PZ, C14, A1, B4); \
      B4  = svdup_n_f32(B[BASE + 9]);	  \
                                          \
      BASE = BASE + DESP;		  \
					  \
      C05 = svmla_f32_z(PZ, C05, A0, B0); \
      C15 = svmla_f32_z(PZ, C15, A1, B0); \
      B0  = svdup_n_f32(B[BASE + 0]);	  \
					  \
      C06 = svmla_f32_z(PZ, C06, A0, B1); \
      C16 = svmla_f32_z(PZ, C16, A1, B1); \
      B1  = svdup_n_f32(B[BASE + 1]);	  \
					  \
      C07 = svmla_f32_z(PZ, C07, A0, B2); \
      C17 = svmla_f32_z(PZ, C17, A1, B2); \
      B2  = svdup_n_f32(B[BASE + 2]);	  \
					  \
      C08 = svmla_f32_z(PZ, C08, A0, B3); \
      C18 = svmla_f32_z(PZ, C18, A1, B3); \
      B3  = svdup_n_f32(B[BASE + 3]);	  \
					  \
      C09 = svmla_f32_z(PZ, C09, A0, B4); \
      C19 = svmla_f32_z(PZ, C19, A1, B4); \
      B4  = svdup_n_f32(B[BASE + 4]);	  \
}


#define GEMM_FMLA_32X10(C00, C01, C02, C03, C04, C05, C06, C07, C08, C09, \
			 C10, C11, C12, C13, C14, C15, C16, C17, C18, C19, \
			 B0,   B1,  B2,  B3,  B4,			\
			 A0, A1, PZ, B, BASE, DESP)			\
{					  \
      C00 = svmla_f32_z(PZ, C00, A0, B0); \
      C10 = svmla_f32_z(PZ, C10, A1, B0); \
      B0  = svdup_n_f32(B[BASE + 5]);	  \
      					  \
      C01 = svmla_f32_z(PZ, C01, A0, B1); \
      C11 = svmla_f32_z(PZ, C11, A1, B1); \
      B1  = svdup_n_f32(B[BASE + 6]);	  \
                                          \
      C02 = svmla_f32_z(PZ, C02, A0, B2); \
      C12 = svmla_f32_z(PZ, C12, A1, B2); \
      B2  = svdup_n_f32(B[BASE + 7]);	  \
					  \
      C03 = svmla_f32_z(PZ, C03, A0, B3); \
      C13 = svmla_f32_z(PZ, C13, A1, B3); \
      B3  = svdup_n_f32(B[BASE + 8]);	  \
					  \
      C04 = svmla_f32_z(PZ, C04, A0, B4); \
      C14 = svmla_f32_z(PZ, C14, A1, B4); \
      B4  = svdup_n_f32(B[BASE + 9]);	  \
					  \
      C05 = svmla_f32_z(PZ, C05, A0, B0); \
      C15 = svmla_f32_z(PZ, C15, A1, B0); \
					  \
      C06 = svmla_f32_z(PZ, C06, A0, B1); \
      C16 = svmla_f32_z(PZ, C16, A1, B1); \
					  \
      C07 = svmla_f32_z(PZ, C07, A0, B2); \
      C17 = svmla_f32_z(PZ, C17, A1, B2); \
                                          \	    
      C08 = svmla_f32_z(PZ, C08, A0, B3); \
      C18 = svmla_f32_z(PZ, C18, A1, B3); \
                                          \
      C09 = svmla_f32_z(PZ, C09, A0, B4); \
      C19 = svmla_f32_z(PZ, C19, A1, B4); \
}


#define ACOL_LOAD_32X10(A0, A1, PZ, A, Ap) \
 { \
      A0 = svld1_f32(PZ, &A[Ap    ]); \
      A1 = svld1_f32(PZ, &A[Ap + 16]); \
 }


inline void ukernel_sve_32x10_fp32( char orderC, int mr, int nr, int kc, float alpha, float *Ar, float *Br, float beta, float *C, int ldC ) {


  if ( kc==0 ) return;
  
  int i, j, k, baseA = 0, baseB = 0, ldCt = MR, Amr, Bnr;

  //+20 Registers for C
  svfloat32_t C00, C01, C02, C03, C04, C05, C06, C07, C08, C09,
              C10, C11, C12, C13, C14, C15, C16, C17, C18, C19;

  //+5 Registers for B. (WARNING: +8 Registers for B, like BLIS GEMM micro-kernel, compiler generates spilling)
  //+1 For Beta FMLAS.
  svfloat32_t B0, B1, B2, B3, B4, B5;

  //+4 Registers for A (2 for prefetch)
  svfloat32_t A0, A1, A2, A3;
  
  svfloat32_t RBeta;
  
  float  zero = 0.0, one = 1.0, *Aptr, *Bptr, Ctmp[MR*NR];
  float Btmp[16];

  int pred = 16;
  if (MR < 16)
    pred = MR;

  svbool_t pred16 = svwhilelt_b32_u32(0, pred);

  Aptr = &Ar[0];
  Bptr = &Br[0];
  Amr  = MR;
  Bnr  = NR;

  //--------------------------------------
  //Load 5 of first [B] row.
  //--------------------------------------
  B0 = svdup_n_f32(Bptr[baseB + 0]); 
  B1 = svdup_n_f32(Bptr[baseB + 1]); 
  B2 = svdup_n_f32(Bptr[baseB + 2]); 
  B3 = svdup_n_f32(Bptr[baseB + 3]); 
  B4 = svdup_n_f32(Bptr[baseB + 4]); 
  //--------------------------------------
  
  //----------------------------------------------
  //[A] Contiguous Load
  //----------------------------------------------
  ACOL_LOAD_32X10(A0, A1, pred16, Aptr, baseA)
  //----------------------------------------------
    
  //-----------------------------------------
  //Prefetching [C]

  //-----------------------------------------
  svprfw(pred16, &Ccol(0, 0), SV_PLDL1STRM);
  svprfw(pred16, &Ccol(0, 1), SV_PLDL1STRM);
  svprfw(pred16, &Ccol(0, 2), SV_PLDL1STRM);
  svprfw(pred16, &Ccol(0, 3), SV_PLDL1STRM);
  svprfw(pred16, &Ccol(0, 4), SV_PLDL1STRM);
  svprfw(pred16, &Ccol(0, 5), SV_PLDL1STRM);
  svprfw(pred16, &Ccol(0, 6), SV_PLDL1STRM);
  svprfw(pred16, &Ccol(0, 7), SV_PLDL1STRM);
  svprfw(pred16, &Ccol(0, 8), SV_PLDL1STRM);
  svprfw(pred16, &Ccol(0, 9), SV_PLDL1STRM);

  svprfw(pred16, &Ccol(16, 0), SV_PLDL1STRM);
  svprfw(pred16, &Ccol(16, 1), SV_PLDL1STRM);
  svprfw(pred16, &Ccol(16, 2), SV_PLDL1STRM);
  svprfw(pred16, &Ccol(16, 3), SV_PLDL1STRM);
  svprfw(pred16, &Ccol(16, 4), SV_PLDL1STRM);
  svprfw(pred16, &Ccol(16, 5), SV_PLDL1STRM);
  svprfw(pred16, &Ccol(16, 6), SV_PLDL1STRM);
  svprfw(pred16, &Ccol(16, 7), SV_PLDL1STRM);
  svprfw(pred16, &Ccol(16, 8), SV_PLDL1STRM);
  svprfw(pred16, &Ccol(16, 9), SV_PLDL1STRM);
  //-----------------------------------------
  
  //------------------------
  //Clear columns
  //------------------------
  C00 = svdup_n_f32(0); //z0
  C01 = svdup_n_f32(0); //z1
  C02 = svdup_n_f32(0); //z2
  C03 = svdup_n_f32(0); //z3
  C04 = svdup_n_f32(0); //z4
  C05 = svdup_n_f32(0); //z5
  C06 = svdup_n_f32(0); //z6
  C07 = svdup_n_f32(0); //z7
  C08 = svdup_n_f32(0); //z7
  C09 = svdup_n_f32(0); //z7
  
  C10 = svdup_n_f32(0); //z10
  C11 = svdup_n_f32(0); //z11
  C12 = svdup_n_f32(0); //z12
  C13 = svdup_n_f32(0); //z13
  C14 = svdup_n_f32(0); //z14
  C15 = svdup_n_f32(0); //z15
  C16 = svdup_n_f32(0); //z16
  C17 = svdup_n_f32(0); //z17
  C18 = svdup_n_f32(0); //z16
  C19 = svdup_n_f32(0); //z17
  //-------------------------

  
  for ( k=0; k < kc - 2; k += 2 ) {
    //---- Iter +1->
    baseA = baseA + MR;      
    ACOL_LOAD_32X10(A2, A3, pred16, Aptr, baseA);
    
    GEMM_FMLA_LD1R_32X10(C00, C01, C02, C03, C04, C05, C06, C07, C08, C09,
			 C10, C11, C12, C13, C14, C15, C16, C17, C18, C19,
			 B0,   B1,  B2,  B3,  B4,  
			 A0,   A1, pred16, Bptr, baseB, NR);            
    //---- Iter +2->
    baseA = baseA + MR;      
    ACOL_LOAD_32X10(A0, A1, pred16, Aptr, baseA);
      
    GEMM_FMLA_LD1R_32X10(C00, C01, C02, C03, C04, C05, C06, C07, C08, C09,
			 C10, C11, C12, C13, C14, C15, C16, C17, C18, C19,
			 B0,   B1,  B2,  B3,  B4,  
			 A2,   A3, pred16, Bptr, baseB, NR);      
  }
  
  
  if (kc % 2 == 0) {
    baseA = baseA + MR;
    
    GEMM_FMLA_LD1R_32X10(C00, C01, C02, C03, C04, C05, C06, C07, C08, C09,
			 C10, C11, C12, C13, C14, C15, C16, C17, C18, C19,
			 B0,   B1,  B2,  B3,  B4,
			 A0,   A1, pred16, Bptr, baseB, NR);
    
    ACOL_LOAD_32X10(A0, A1, pred16, Aptr, baseA);      
  }
  
  GEMM_FMLA_32X10(C00, C01, C02, C03, C04, C05, C06, C07, C08, C09,
		  C10, C11, C12, C13, C14, C15, C16, C17, C18, C19,
		  B0,   B1,  B2,  B3,  B4,
		  A0,   A1, pred16, Bptr, baseB, NR);
  //}

  
  if ( (mr<MR) || (nr<NR) ) {    
    
    svst1_f32(pred16, &Ctref(0, 0),  C00);
    svst1_f32(pred16, &Ctref(0, 1),  C01);
    svst1_f32(pred16, &Ctref(0, 2),  C02);
    svst1_f32(pred16, &Ctref(0, 3),  C03);
    svst1_f32(pred16, &Ctref(0, 4),  C04);
    svst1_f32(pred16, &Ctref(0, 5),  C05);
    svst1_f32(pred16, &Ctref(0, 6),  C06);
    svst1_f32(pred16, &Ctref(0, 7),  C07);
    svst1_f32(pred16, &Ctref(0, 8),  C08);
    svst1_f32(pred16, &Ctref(0, 9),  C09);

    svst1_f32(pred16, &Ctref(16, 0),  C10);
    svst1_f32(pred16, &Ctref(16, 1),  C11);
    svst1_f32(pred16, &Ctref(16, 2),  C12);
    svst1_f32(pred16, &Ctref(16, 3),  C13);
    svst1_f32(pred16, &Ctref(16, 4),  C14);
    svst1_f32(pred16, &Ctref(16, 5),  C15);
    svst1_f32(pred16, &Ctref(16, 6),  C16);
    svst1_f32(pred16, &Ctref(16, 7),  C17);
    svst1_f32(pred16, &Ctref(16, 8),  C18);
    svst1_f32(pred16, &Ctref(16, 9),  C19);

    if ( beta != zero ) {
      for ( j=0; j < nr; j++ )
	for ( i=0; i < mr; i++ )
	  Ccol(i,j) = Ccol(i,j) + Ctcol(i,j);
    } else {
      for ( j=0; j<nr; j++ )
	for ( i=0; i<mr; i++ )
	  Ccol(i,j) = Ctcol(i,j);
    }
    
  } else {
    if ( beta != zero ) {
      RBeta = svdup_n_f32(beta);
      
      B0 = svld1_f32(pred16, &Ccol(0, 0));
      B1 = svld1_f32(pred16, &Ccol(0, 1));
      B2 = svld1_f32(pred16, &Ccol(0, 2));
      B3 = svld1_f32(pred16, &Ccol(0, 3));
      B4 = svld1_f32(pred16, &Ccol(0, 4));
      B5 = svld1_f32(pred16, &Ccol(0, 5));
      A0 = svld1_f32(pred16, &Ccol(0, 6));
      A1 = svld1_f32(pred16, &Ccol(0, 7));
      A2 = svld1_f32(pred16, &Ccol(0, 8));
      A3 = svld1_f32(pred16, &Ccol(0, 9));
      
      C00 = svmla_f32_z(pred16, C00, RBeta, B0);
      C01 = svmla_f32_z(pred16, C01, RBeta, B1);
      C02 = svmla_f32_z(pred16, C02, RBeta, B2);
      C03 = svmla_f32_z(pred16, C03, RBeta, B3);
      C04 = svmla_f32_z(pred16, C04, RBeta, B4);
      C05 = svmla_f32_z(pred16, C05, RBeta, B5);
      C06 = svmla_f32_z(pred16, C06, RBeta, A0);
      C07 = svmla_f32_z(pred16, C07, RBeta, A1);
      C08 = svmla_f32_z(pred16, C08, RBeta, A2);
      C09 = svmla_f32_z(pred16, C09, RBeta, A3);
      
      B0 = svld1_f32(pred16, &Ccol(16, 0));
      B1 = svld1_f32(pred16, &Ccol(16, 1));
      B2 = svld1_f32(pred16, &Ccol(16, 2));
      B3 = svld1_f32(pred16, &Ccol(16, 3));
      B4 = svld1_f32(pred16, &Ccol(16, 4));
      B5 = svld1_f32(pred16, &Ccol(16, 5));
      A0 = svld1_f32(pred16, &Ccol(16, 6));
      A1 = svld1_f32(pred16, &Ccol(16, 7));
      A2 = svld1_f32(pred16, &Ccol(16, 8));
      A3 = svld1_f32(pred16, &Ccol(16, 9));
      
      C10 = svmla_f32_z(pred16, C10, RBeta, B0);
      C11 = svmla_f32_z(pred16, C11, RBeta, B1);
      C12 = svmla_f32_z(pred16, C12, RBeta, B2);
      C13 = svmla_f32_z(pred16, C13, RBeta, B3);
      C14 = svmla_f32_z(pred16, C14, RBeta, B4);
      C15 = svmla_f32_z(pred16, C15, RBeta, B5);
      C16 = svmla_f32_z(pred16, C16, RBeta, A0);
      C17 = svmla_f32_z(pred16, C17, RBeta, A1);
      C18 = svmla_f32_z(pred16, C18, RBeta, A2);
      C19 = svmla_f32_z(pred16, C19, RBeta, A3);
    }
    
    svst1_f32(pred16, &Ccol(0, 0),  C00);
    svst1_f32(pred16, &Ccol(0, 1),  C01);
    svst1_f32(pred16, &Ccol(0, 2),  C02);
    svst1_f32(pred16, &Ccol(0, 3),  C03);
    svst1_f32(pred16, &Ccol(0, 4),  C04);
    svst1_f32(pred16, &Ccol(0, 5),  C05);
    svst1_f32(pred16, &Ccol(0, 6),  C06);
    svst1_f32(pred16, &Ccol(0, 7),  C07);
    svst1_f32(pred16, &Ccol(0, 8),  C08);
    svst1_f32(pred16, &Ccol(0, 9),  C09);

    svst1_f32(pred16, &Ccol(16, 0),  C10);
    svst1_f32(pred16, &Ccol(16, 1),  C11);
    svst1_f32(pred16, &Ccol(16, 2),  C12);
    svst1_f32(pred16, &Ccol(16, 3),  C13);
    svst1_f32(pred16, &Ccol(16, 4),  C14);
    svst1_f32(pred16, &Ccol(16, 5),  C15);
    svst1_f32(pred16, &Ccol(16, 6),  C16);
    svst1_f32(pred16, &Ccol(16, 7),  C17);
    svst1_f32(pred16, &Ccol(16, 8),  C18);
    svst1_f32(pred16, &Ccol(16, 9),  C19);
  } 

}


//======================================================================================
//**************************************************************************************
//*                       M I C R O - K E R N E L    32 X 8                            *
//**************************************************************************************
//======================================================================================


#define GEMM_FMLA_LD1R_32X8(C00, C01, C02, C03, C04, C05, C06, C07, \
			    C10, C11, C12, C13, C14, C15, C16, C17, \
			    B0,   B1,  B2,  B3,  B4,  B5,  B6,  B7, \
			    A0, A1, PZ, B, Bp)			    \
 { \
      C00 = svmla_f32_z(PZ, C00, A0, B0); \
      C10 = svmla_f32_z(PZ, C10, A1, B0); \
      B0  = svdup_n_f32(B[Bp + 0]);\
 \
      C01 = svmla_f32_z(PZ, C01, A0, B1); \
      C11 = svmla_f32_z(PZ, C11, A1, B1); \
      B1  = svdup_n_f32(B[Bp + 1]); \
      C02 = svmla_f32_z(PZ, C02, A0, B2); \
      C12 = svmla_f32_z(PZ, C12, A1, B2); \
      B2  = svdup_n_f32(B[Bp + 2]); \
 \
      C03 = svmla_f32_z(PZ, C03, A0, B3); \
      C13 = svmla_f32_z(PZ, C13, A1, B3); \
      B3  = svdup_n_f32(B[Bp + 3]); \
 \
      C04 = svmla_f32_z(PZ, C04, A0, B4); \
      C14 = svmla_f32_z(PZ, C14, A1, B4); \
      B4  = svdup_n_f32(B[Bp + 4]); \
 \
      C05 = svmla_f32_z(PZ, C05, A0, B5); \
      C15 = svmla_f32_z(PZ, C15, A1, B5); \
      B5  = svdup_n_f32(B[Bp + 5]); \
 \
      C06 = svmla_f32_z(PZ, C06, A0, B6); \
      C16 = svmla_f32_z(PZ, C16, A1, B6); \
      B6  = svdup_n_f32(B[Bp + 6]); \
 \
      C07 = svmla_f32_z(PZ, C07, A0, B7); \
      C17 = svmla_f32_z(PZ, C17, A1, B7); \
      B7  = svdup_n_f32(B[Bp + 7]); \
 }

#define GEMM_FMLA_32X8(C00, C01, C02, C03, C04, C05, C06, C07, \
		       C10, C11, C12, C13, C14, C15, C16, C17, \
		       B0,   B1,  B2,  B3,  B4,  B5,  B6,  B7, \
		       A0, A1, PZ)			       \
 { \
      C00 = svmla_f32_z(PZ, C00, A0, B0); \
      C10 = svmla_f32_z(PZ, C10, A1, B0); \
 \
      C01 = svmla_f32_z(PZ, C01, A0, B1); \
      C11 = svmla_f32_z(PZ, C11, A1, B1); \
 \
      C02 = svmla_f32_z(PZ, C02, A0, B2); \
      C12 = svmla_f32_z(PZ, C12, A1, B2); \
 \
      C03 = svmla_f32_z(PZ, C03, A0, B3); \
      C13 = svmla_f32_z(PZ, C13, A1, B3); \
 \
      C04 = svmla_f32_z(PZ, C04, A0, B4); \
      C14 = svmla_f32_z(PZ, C14, A1, B4); \
 \
      C05 = svmla_f32_z(PZ, C05, A0, B5); \
      C15 = svmla_f32_z(PZ, C15, A1, B5); \
 \
      C06 = svmla_f32_z(PZ, C06, A0, B6); \
      C16 = svmla_f32_z(PZ, C16, A1, B6); \
 \
      C07 = svmla_f32_z(PZ, C07, A0, B7); \
      C17 = svmla_f32_z(PZ, C17, A1, B7); \
 }


#define ACOL_LOAD_32X8(A0, A1, PZ, A, Ap) \
 { \
      A0 = svld1_f32(PZ, &A[Ap    ]); \
      A1 = svld1_f32(PZ, &A[Ap + 16]); \
 }

inline void ukernel_sve_32x8_fp32( char orderC, int mr, int nr, int kc, float alpha, float *Ar, float *Br, float beta, float *C, int ldC ) {


  if ( kc==0 ) return;
  
  int i, j, k, baseA = 0, baseB = 0, ldCt = MR, Amr, Bnr;

  //+16 Registers for C
  svfloat32_t C00, C01, C02, C03, C04, C05, C06, C07,
              C10, C11, C12, C13, C14, C15, C16, C17;

  //+8 Registers for B
  svfloat32_t B0, B1, B2, B3, B4, B5, B6, B7;

  //+4 Registers for A (2 for prefetch)
  svfloat32_t A0, A1, A2, A3;
  
  svfloat32_t RBeta;
  
  float  zero = 0.0, one = 1.0, *Aptr, *Bptr, Ctmp[MR*NR];
  float Btmp[16];
  
  svbool_t pred16 = svwhilelt_b32_u32(0, 16);

  Aptr = &Ar[0];
  Bptr = &Br[0];
  Amr  = MR;
  Bnr  = NR;
  
  //--------------------------------------
  //Load 8/10 of first [B] row.
  //--------------------------------------
  B0 = svdup_n_f32(Bptr[baseB + 0]); //z20
  B1 = svdup_n_f32(Bptr[baseB + 1]); //z21
  B2 = svdup_n_f32(Bptr[baseB + 2]); //z22
  B3 = svdup_n_f32(Bptr[baseB + 3]); //z23
  B4 = svdup_n_f32(Bptr[baseB + 4]); //z24
  B5 = svdup_n_f32(Bptr[baseB + 5]); //z25
  B6 = svdup_n_f32(Bptr[baseB + 6]); //z26
  B7 = svdup_n_f32(Bptr[baseB + 7]); //z27
  //--------------------------------------
  
  //----------------------------------------------
  //[A] Contiguous Load
  //----------------------------------------------
  ACOL_LOAD_32X8(A0, A1, pred16, Aptr, baseA)
  //----------------------------------------------
  
  //-----------------------------------------
  //Prefetching [C]
  //-----------------------------------------

    /*svprfw(pred16, &Ccol(0, 0), SV_PLDL1STRM);
  svprfw(pred16, &Ccol(0, 1), SV_PLDL1STRM);
  svprfw(pred16, &Ccol(0, 2), SV_PLDL1STRM);
  svprfw(pred16, &Ccol(0, 3), SV_PLDL1STRM);
  svprfw(pred16, &Ccol(0, 4), SV_PLDL1STRM);
  svprfw(pred16, &Ccol(0, 5), SV_PLDL1STRM);
  svprfw(pred16, &Ccol(0, 6), SV_PLDL1STRM);
  svprfw(pred16, &Ccol(0, 7), SV_PLDL1STRM);

  svprfw(pred16, &Ccol(16, 0), SV_PLDL1STRM);
  svprfw(pred16, &Ccol(16, 1), SV_PLDL1STRM);
  svprfw(pred16, &Ccol(16, 2), SV_PLDL1STRM);
  svprfw(pred16, &Ccol(16, 3), SV_PLDL1STRM);
  svprfw(pred16, &Ccol(16, 4), SV_PLDL1STRM);
  svprfw(pred16, &Ccol(16, 5), SV_PLDL1STRM);
  svprfw(pred16, &Ccol(16, 6), SV_PLDL1STRM);
  svprfw(pred16, &Ccol(16, 7), SV_PLDL1STRM);*/
  //-----------------------------------------
  
  //------------------------
  //Clear columns
  //------------------------
  C00 = svdup_n_f32(0); //z0
  C01 = svdup_n_f32(0); //z1
  C02 = svdup_n_f32(0); //z2
  C03 = svdup_n_f32(0); //z3
  C04 = svdup_n_f32(0); //z4
  C05 = svdup_n_f32(0); //z5
  C06 = svdup_n_f32(0); //z6
  C07 = svdup_n_f32(0); //z7
  
  C10 = svdup_n_f32(0); //z10
  C11 = svdup_n_f32(0); //z11
  C12 = svdup_n_f32(0); //z12
  C13 = svdup_n_f32(0); //z13
  C14 = svdup_n_f32(0); //z14
  C15 = svdup_n_f32(0); //z15
  C16 = svdup_n_f32(0); //z16
  C17 = svdup_n_f32(0); //z17
  //-------------------------

  if (alpha != zero) {    
    for ( k=0; k < kc - 2; k += 2 ) {      
      //---- Iter +1->
      baseA = baseA + MR;
      baseB = baseB + NR;
      
      ACOL_LOAD_32X8(A2, A3, pred16, Aptr, baseA);

      GEMM_FMLA_LD1R_32X8(C00, C01, C02, C03, C04, C05, C06, C07,
			  C10, C11, C12, C13, C14, C15, C16, C17,
			  B0,   B1,  B2,  B3,  B4,  B5,  B6,  B7,
			  A0,   A1, pred16, Bptr, baseB);
            
      //---- Iter +2->
      baseA = baseA + MR;
      baseB = baseB + NR;
      
      ACOL_LOAD_32X8(A0, A1, pred16, Aptr, baseA);

      GEMM_FMLA_LD1R_32X8(C00, C01, C02, C03, C04, C05, C06, C07,
			  C10, C11, C12, C13, C14, C15, C16, C17,
			  B0,   B1,  B2,  B3,  B4,  B5,  B6,  B7,
			  A2,   A3, pred16, Bptr, baseB);      
    }
	  
    if (kc % 2 == 0) {
      baseB = baseB + NR;
      baseA = baseA + MR;
      
      GEMM_FMLA_LD1R_32X8(C00, C01, C02, C03, C04, C05, C06, C07,
			  C10, C11, C12, C13, C14, C15, C16, C17,
			  B0,   B1,  B2,  B3,  B4,  B5,  B6,  B7,
			  A0,   A1, pred16, Bptr, baseB);
      
      ACOL_LOAD_32X8(A0, A1, pred16, Aptr, baseA);      
    }
    
    GEMM_FMLA_32X8(C00, C01, C02, C03, C04, C05, C06, C07,
		   C10, C11, C12, C13, C14, C15, C16, C17,
		   B0,   B1,  B2,  B3,  B4,  B5,  B6,  B7,
		   A0,   A1, pred16);    
  }
  
  if ( (mr<MR) || (nr<NR) ) {    
    
    svst1_f32(pred16, &Ctref(0, 0),  C00);
    svst1_f32(pred16, &Ctref(0, 1),  C01);
    svst1_f32(pred16, &Ctref(0, 2),  C02);
    svst1_f32(pred16, &Ctref(0, 3),  C03);
    svst1_f32(pred16, &Ctref(0, 4),  C04);
    svst1_f32(pred16, &Ctref(0, 5),  C05);
    svst1_f32(pred16, &Ctref(0, 6),  C06);
    svst1_f32(pred16, &Ctref(0, 7),  C07);

    svst1_f32(pred16, &Ctref(16, 0),  C10);
    svst1_f32(pred16, &Ctref(16, 1),  C11);
    svst1_f32(pred16, &Ctref(16, 2),  C12);
    svst1_f32(pred16, &Ctref(16, 3),  C13);
    svst1_f32(pred16, &Ctref(16, 4),  C14);
    svst1_f32(pred16, &Ctref(16, 5),  C15);
    svst1_f32(pred16, &Ctref(16, 6),  C16);
    svst1_f32(pred16, &Ctref(16, 7),  C17);

    if ( beta != zero ) {
      for ( j=0; j < nr; j++ )
	for ( i=0; i < mr; i++ )
	  Ccol(i,j) = beta*Ccol(i,j) + Ctcol(i,j);
    } else {
    	for ( j=0; j<nr; j++ )
    	  for ( i=0; i<mr; i++ )
    	    Ccol(i,j) = Ctcol(i,j);
    }
    
  } else {
    if ( beta != zero ) {
      RBeta = svdup_n_f32(beta);
      
      B0 = svld1_f32(pred16, &Ccol(0, 0));
      B1 = svld1_f32(pred16, &Ccol(0, 1));
      B2 = svld1_f32(pred16, &Ccol(0, 2));
      B3 = svld1_f32(pred16, &Ccol(0, 3));
      B4 = svld1_f32(pred16, &Ccol(0, 4));
      B5 = svld1_f32(pred16, &Ccol(0, 5));
      B6 = svld1_f32(pred16, &Ccol(0, 6));
      B7 = svld1_f32(pred16, &Ccol(0, 7));
      
      C00 = svmla_f32_z(pred16, C00, RBeta, B0);
      C01 = svmla_f32_z(pred16, C01, RBeta, B1);
      C02 = svmla_f32_z(pred16, C02, RBeta, B2);
      C03 = svmla_f32_z(pred16, C03, RBeta, B3);
      C04 = svmla_f32_z(pred16, C04, RBeta, B4);
      C05 = svmla_f32_z(pred16, C05, RBeta, B5);
      C06 = svmla_f32_z(pred16, C06, RBeta, B6);
      C07 = svmla_f32_z(pred16, C07, RBeta, B7);
      
      B0 = svld1_f32(pred16, &Ccol(16, 0));
      B1 = svld1_f32(pred16, &Ccol(16, 1));
      B2 = svld1_f32(pred16, &Ccol(16, 2));
      B3 = svld1_f32(pred16, &Ccol(16, 3));
      B4 = svld1_f32(pred16, &Ccol(16, 4));
      B5 = svld1_f32(pred16, &Ccol(16, 5));
      B6 = svld1_f32(pred16, &Ccol(16, 6));
      B7 = svld1_f32(pred16, &Ccol(16, 7));
      
      C10 = svmla_f32_z(pred16, C10, RBeta, B0);
      C11 = svmla_f32_z(pred16, C11, RBeta, B1);
      C12 = svmla_f32_z(pred16, C12, RBeta, B2);
      C13 = svmla_f32_z(pred16, C13, RBeta, B3);
      C14 = svmla_f32_z(pred16, C14, RBeta, B4);
      C15 = svmla_f32_z(pred16, C15, RBeta, B5);
      C16 = svmla_f32_z(pred16, C16, RBeta, B6);
      C17 = svmla_f32_z(pred16, C17, RBeta, B7);
    }
    
    svst1_f32(pred16, &Ccol(0, 0),  C00);
    svst1_f32(pred16, &Ccol(0, 1),  C01);
    svst1_f32(pred16, &Ccol(0, 2),  C02);
    svst1_f32(pred16, &Ccol(0, 3),  C03);
    svst1_f32(pred16, &Ccol(0, 4),  C04);
    svst1_f32(pred16, &Ccol(0, 5),  C05);
    svst1_f32(pred16, &Ccol(0, 6),  C06);
    svst1_f32(pred16, &Ccol(0, 7),  C07);

    svst1_f32(pred16, &Ccol(16, 0),  C10);
    svst1_f32(pred16, &Ccol(16, 1),  C11);
    svst1_f32(pred16, &Ccol(16, 2),  C12);
    svst1_f32(pred16, &Ccol(16, 3),  C13);
    svst1_f32(pred16, &Ccol(16, 4),  C14);
    svst1_f32(pred16, &Ccol(16, 5),  C15);
    svst1_f32(pred16, &Ccol(16, 6),  C16);
    svst1_f32(pred16, &Ccol(16, 7),  C17);

  } 

}


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

   author    = "Héctor Martínez Pérez"
   contact   = "el2mapeh@uco.es"
   license   = "GPLv3"
   status    = "Production"
   version   = "1.1"
*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <emmintrin.h>

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))

#define Acol(a1,a2)  A[ (a2)*(ldA)+(a1) ]
#define Bcol(a1,a2)  B[ (a2)*(ldB)+(a1) ]
#define Ccol(a1,a2)  C[ (a2)*(ldC)+(a1) ]
#define Mcol(a1,a2)  M[ (a2)*(ldM)+(a1) ]

#define Arow(a1,a2)  A[ (a1)*(ldA)+(a2) ]
#define Brow(a1,a2)  B[ (a1)*(ldB)+(a2) ]
#define Crow(a1,a2)  C[ (a1)*(ldC)+(a2) ]
#define Mrow(a1,a2)  M[ (a1)*(ldM)+(a2) ]
#define Ctref(a1,a2) Ctmp[ (a2)*(ldCt)+(a1) ]
#define Ctrow(a1,a2) Ctmp[ (a1)*(ldCt)+(a2) ]

void pack_RB( int mc, int nc, float *M, int ldM, float *Mc, int RR );
void pack_CB( int mc, int nc, float *M, int ldM, float *Mc, int RR );
void gemm_base_Cresident( int m, int n, int k, float alpha, float *A, int ldA, float *B, int ldB, float beta,  float *C, int ldC );
void gemm_base_no_packing( int m, int n, int k, float alpha, float *A, int ldA, float *B, int ldB, float beta,  float *C, int ldC );
void gemm_microkernel_Cresident_neon_4x4_fp32 ( int mr, int nr, int kc, float alpha, float *Ar, float *Br, float beta, float      *C, int ldC );
void gemm_microkernel_Cresident_neon_8x8_fp32 ( int mr, int nr, int kc, float alpha, float *Ar, float *Br, float beta, float      *C, int ldC );
void ukernel_neon_4x4_np_fp32( int mr, int nr, int kc, float alpha, float *Ar, size_t ldA, float *Br, size_t ldB, float beta, float *C, int ldC );

void gemm_block_ukernel_8x8( size_t m, size_t n, size_t k, float alpha, float *A, size_t ldA, float *B, size_t ldB, float beta, float *C, size_t ldC, float *Ac, float *Bc, size_t MC, size_t NC, size_t KC ) {

  int MR=8;
  int NR=8;
  size_t    ic, jc, pc, mc, nc, kc, ir, jr, mr, nr; 
  float  zero = 0.0, one = 1.0, betaI; 
  float  *Aptr, *Bptr, *Cptr;
  
  Ac = A;

  // Quick return if possible
  if ( (m==0)||(n==0)||(((alpha==zero)||(k==0))&&(beta==one)) )
    return;

  for ( jc=0; jc<n; jc+=NC ) {
    nc = min(n-jc, NC); 

    for ( pc=0; pc<k; pc+=KC ) {
      kc = min(k-pc, KC); 
      
        Bptr = &Brow(pc,jc);
      
      pack_CB( kc, nc, Bptr, ldB, Bc, NR);
      
      if ( pc==0 )
        betaI = beta;
      else
        betaI = one;
      
      for ( ic=0; ic<m; ic+=MC ) {
        mc = min(m-ic, MC); 
	
        //Aptr = &Arow(ic, pc);
	
        //pack_RB( mc, kc, Aptr, ldA, Ac, MR);
	
        for ( jr=0; jr<nc; jr+=NR ) {
          nr = min(nc-jr, NR); 
	  
          for ( ir=0; ir<mc; ir+=MR ) {
            mr = min(mc-ir, MR); 
	    
            Cptr = &Crow(ic+ir,jc+jr);
            gemm_microkernel_Cresident_neon_8x8_fp32( mr, nr, kc, alpha, &Ac[ir*kc], &Bc[jr*kc], betaI, Cptr, ldC );
	  }
	  
        }
      }
    }
  }
}

void gemm_block_ukernel_max( size_t m, size_t n, size_t k, float alpha, float *A, size_t ldA, float *B, size_t ldB, float beta, float *C, size_t ldC, float *Ac, float *Bc, size_t MC, size_t NC, size_t KC ) {

  int MR=4;
  int NR=4;
  size_t    ic, jc, pc, mc, nc, kc, ir, jr, mr, nr; 
  float  zero = 0.0, one = 1.0, betaI; 
  float  *Aptr, *Bptr, *Cptr;

  // Quick return if possible
  if ( (m==0)||(n==0)||(((alpha==zero)||(k==0))&&(beta==one)) )
    return;

  for ( jc=0; jc<n; jc+=NC ) {
    nc = min(n-jc, NC); 

    for ( pc=0; pc<k; pc+=KC ) {
      kc = min(k-pc, KC); 
      
        Bptr = &Brow(pc,jc);
      
      pack_CB( kc, nc, Bptr, ldB, Bc, NR);
      
      if ( pc==0 )
        betaI = beta;
      else
        betaI = one;
      
      for ( ic=0; ic<m; ic+=MC ) {
        mc = min(m-ic, MC); 
	
          Aptr = &Arow(ic, pc);
	
        pack_RB( mc, kc, Aptr, ldA, Ac, MR);
	
	//#pragma omp  parallel for private(nr, ir, mr, Cptr) firstprivate (Ac, Bc)   
        for ( jr=0; jr<nc; jr+=NR ) {
          nr = min(nc-jr, NR); 
	  
          for ( ir=0; ir<mc; ir+=MR ) {
            mr = min(mc-ir, MR); 
	    
            Cptr = &Crow(ic+ir,jc+jr);
	    //gemm_base_Cresident( mr, nr, kc, alpha, &Ac[ir*kc], MR, &Bc[jr*kc], NR, betaI, Cptr, ldC );
            gemm_microkernel_Cresident_neon_4x4_fp32( mr, nr, kc, alpha, &Ac[ir*kc], &Bc[jr*kc], betaI, Cptr, ldC );
	  }
	  
        }
      }
    }
  }
}


void gemm_block_ukernel( size_t m, size_t n, size_t k, float alpha, float *A, size_t ldA, float *B, size_t ldB, float beta, float *C, size_t ldC, float *Ac, float *Bc, size_t MC, size_t NC, size_t KC ) {

  int MR=4;
  int NR=4;
  size_t    ic, jc, pc, mc, nc, kc, ir, jr, mr, nr; 
  float  zero = 0.0, one = 1.0, betaI; 
  float  *Aptr, *Bptr, *Cptr;

  // Quick return if possible
  if ( (m==0)||(n==0)||(((alpha==zero)||(k==0))&&(beta==one)) )
    return;

  for ( jc=0; jc<n; jc+=NC ) {
    nc = min(n-jc, NC); 

    for ( pc=0; pc<k; pc+=KC ) {
      kc = min(k-pc, KC); 
      
        Bptr = &Brow(pc,jc);
      
      pack_CB( kc, nc, Bptr, ldB, Bc, NR);
      
      if ( pc==0 )
        betaI = beta;
      else
        betaI = one;
      
      for ( ic=0; ic<m; ic+=MC ) {
        mc = min(m-ic, MC); 
	
          Aptr = &Arow(ic, pc);
	
        pack_RB( mc, kc, Aptr, ldA, Ac, MR);
	
	//#pragma omp  parallel for private(nr, ir, mr, Cptr) firstprivate (Ac, Bc)   
        for ( jr=0; jr<nc; jr+=NR ) {
          nr = min(nc-jr, NR); 
	  
          for ( ir=0; ir<mc; ir+=MR ) {
            mr = min(mc-ir, MR); 
	    
            Cptr = &Crow(ic+ir,jc+jr);
	    //gemm_base_Cresident( mr, nr, kc, alpha, &Ac[ir*kc], MR, &Bc[jr*kc], NR, betaI, Cptr, ldC );
            gemm_microkernel_Cresident_neon_4x4_fp32( mr, nr, kc, alpha, &Ac[ir*kc], &Bc[jr*kc], betaI, Cptr, ldC );
	  }
	  
        }
      }
    }
  }
}

void gemm_block_no_packing( size_t m, size_t n, size_t k, float alpha, float *A, size_t ldA, float *B, size_t ldB, float beta, float *C, size_t ldC, float *Ac, float *Bc, size_t MC, size_t NC, size_t KC ) {

  size_t    ic, jc, pc, mc, nc, kc, ir, jr, mr, nr; 
  float  zero = 0.0, one = 1.0, betaI; 
  float  *Aptr, *Bptr, *Cptr;

  int MR=4;
  int NR=4;
  // Quick return if possible
  if ( (m==0)||(n==0)||(((alpha==zero)||(k==0))&&(beta==one)) )
    return;

  for ( jc=0; jc<n; jc+=NC ) {
    nc = min(n-jc, NC); 

    for ( pc=0; pc<k; pc+=KC ) {
      kc = min(k-pc, KC); 
      
        Bptr = &Brow(pc,jc);
      
      //pack_CB( kc, nc, Bptr, ldB, Bc, NR);
      
      if ( pc==0 )
        betaI = beta;
      else
        betaI = one;
      
      for ( ic=0; ic<m; ic+=MC ) {
        mc = min(m-ic, MC); 
	
        Aptr = &Arow(ic, pc);
	
        //pack_RB( mc, kc, Aptr, ldA, Ac, MR);
	
	//#pragma omp  parallel for private(nr, ir, mr, Cptr) firstprivate (Ac, Bc)   
        for ( jr=0; jr<nc; jr+=NR ) {
          nr = min(nc-jr, NR); 
	  
          for ( ir=0; ir<mc; ir+=MR ) {
            mr = min(mc-ir, MR); 
	    
            Cptr = &Crow(ic+ir,jc+jr);
	    //No packing [A]
	    //gemm_base_no_packing( mr, nr, kc, alpha, &Arow(ic + ir, pc), ldA, &Bc[jr*kc], NR, betaI, Cptr, ldC );
	    //No packing [A & B]
	    //gemm_base_no_packing( mr, nr, kc, alpha, &Arow(ic + ir, pc), ldA, &Brow(pc, jc + jr), ldB, betaI, Cptr, ldC );
            ukernel_neon_4x4_np_fp32( mr, nr, kc, alpha, &Arow(ic + ir, pc), ldA, &Brow(pc, jc + jr), ldB, betaI, Cptr, ldC);
	  }
	  
        }
      }
    }
  }
}

void gemm_base_no_packing( int m, int n, int k, float alpha, float *A, int ldA, 
                          float *B, int ldB, float beta,  float *C, int ldC ){
  int    i, j, p;
  float  zero = 0.0, tmp;

  for ( j=0; j<n; j++ )
    for ( i=0; i<m; i++ ) {
      tmp = 0.0; 
      for ( p=0; p<k; p++ ) 
        tmp += Arow(i,p) * Brow(p,j);

      if ( beta==zero ) {
          Crow(i,j) = alpha*tmp;
      }
      else {
          Crow(i,j) = alpha*tmp + beta*Crow(i,j);
      }
    }
}

void pack_RB( int mc, int nc, float *M, int ldM, float *Mc, int RR ){
  int    i, j, ii, k, rr;

    for ( i=0; i<mc; i+=RR ) { 
      k = i*nc;
      rr = min( mc-i, RR );
      for ( j=0; j<nc; j++ ) {
        for ( ii=0; ii<rr; ii++ ) {
           Mc[k] = Mcol(j,i+ii);
          k++;
        }
        k += (RR-rr);
      }
    }
}

void pack_CB( int mc, int nc, float *M, int ldM, float *Mc, int RR ){
  int    i, j, jj, k, nr;
    for ( j=0; j<nc; j+=RR ) { 
      k = j*mc;
      nr = min( nc-j, RR );
      for ( i=0; i<mc; i++ ) {
        for ( jj=0; jj<nr; jj++ ) {
          Mc[k] = Mcol(j+jj,i);
          k++;
        }
        k += (RR-nr);
      }
    }
}

void gemm_base_Cresident( int m, int n, int k, 
                          float alpha, float *A, int ldA, 
                                       float *B, int ldB, 
                          float beta,  float *C, int ldC ){
  int    i, j, p;
  float  zero = 0.0, tmp;

  for ( j=0; j<n; j++ )
    for ( i=0; i<m; i++ ) {
      tmp = 0.0; 
      for ( p=0; p<k; p++ ) 
        tmp += Acol(i,p) * Brow(p,j);

      if ( beta==zero ) {
          Crow(i,j) = alpha*tmp;
      }
      else {
          Crow(i,j) = alpha*tmp + beta*Crow(i,j);
      }
    }
}

inline void gemm_microkernel_Cresident_neon_8x8_fp32( int mr, int nr, int kc, float alpha, float *Ar, float *Br, float beta, float *C, int ldC ){

    __m128   C00, C01, C02, C03, C04, C05, C06, C07,
             C10, C11, C12, C13, C14, C15, C16, C17;

    __m128   L00, L01, L02, L03, L04, L05, L06, L07,
             L10, L11, L12, L13, L14, L15, L16, L17,
             B0, A0, B1, B2, B3, B4, A1, B5, B6, B7 ; 

  int MR=8;
  int NR=8;

    int      i, j, k, baseA = 0, baseB = 0, ldCt = MR, Amr, Bnr;
    float  zero = 0.0, one = 1.0, *Aptr, *Bptr, Ctmp[MR*NR];
    
    if ( kc==0 ) return;

    C00 = _mm_set1_ps(0); 
    C01 = _mm_set1_ps(0); 
    C02 = _mm_set1_ps(0); 
    C03 = _mm_set1_ps(0); 
    C04 = _mm_set1_ps(0); 
    C05 = _mm_set1_ps(0); 
    C06 = _mm_set1_ps(0); 
    C07 = _mm_set1_ps(0); 
    C10 = _mm_set1_ps(0); 
    C11 = _mm_set1_ps(0); 
    C12 = _mm_set1_ps(0); 
    C13 = _mm_set1_ps(0); 
    C14 = _mm_set1_ps(0); 
    C15 = _mm_set1_ps(0); 
    C16 = _mm_set1_ps(0); 
    C17 = _mm_set1_ps(0); 

    Aptr = &Br[0];
    Bptr = &Ar[0];
    Amr  = NR;
    Bnr  = MR;
    
    for ( k=0; k<kc; k++ ) {
		    
      A0 = _mm_load_ps(&Aptr[baseA]);
      A1 = _mm_load_ps(&Aptr[baseA + 4]);

      //B0 = _mm_load_ps(&Bptr[baseB]);
      B0 = _mm_set1_ps(Bptr[baseB]);
      B1 = _mm_set1_ps(Bptr[baseB+1]);
      B2 = _mm_set1_ps(Bptr[baseB+2]);
      B3 = _mm_set1_ps(Bptr[baseB+3]);
      
      //B1 = _mm_load_ps(&Bptr[baseB + 4]);
      B4 = _mm_set1_ps(Bptr[baseB+4]);
      B5 = _mm_set1_ps(Bptr[baseB+5]);
      B6 = _mm_set1_ps(Bptr[baseB+6]);
      B7 = _mm_set1_ps(Bptr[baseB+7]);

		    
      C00  += A0 * B0; 
      C01  += A0 * B1; 
      C02  += A0 * B2; 
      C03  += A0 * B3; 
      C04  += A0 * B4; 
      C05  += A0 * B5; 
      C06  += A0 * B6; 
      C07  += A0 * B7; 
      
      C10 += A1 * B0; 
      C11 += A1 * B1; 
      C12 += A1 * B2; 
      C13 += A1 * B3; 
      C14 += A1 * B4; 
      C15 += A1 * B5; 
      C16 += A1 * B6; 
      C17 += A1 * B7; 
		    
      baseA = baseA+Amr; 
      baseB = baseB+Bnr;
    }
	    
    if ( (mr<MR)||(nr<NR) ) {
      _mm_store_ps(&Ctref(0,0), C00);
      _mm_store_ps(&Ctref(0,1), C01);
      _mm_store_ps(&Ctref(0,2), C02);
      _mm_store_ps(&Ctref(0,3), C03);
      _mm_store_ps(&Ctref(0,4), C04);
      _mm_store_ps(&Ctref(0,5), C05);
      _mm_store_ps(&Ctref(0,6), C06);
      _mm_store_ps(&Ctref(0,7), C07);
	    
      _mm_store_ps(&Ctref(4,0), C10);
      _mm_store_ps(&Ctref(4,1), C11);
      _mm_store_ps(&Ctref(4,2), C12);
      _mm_store_ps(&Ctref(4,3), C13);
      _mm_store_ps(&Ctref(4,4), C14);
      _mm_store_ps(&Ctref(4,5), C15);
      _mm_store_ps(&Ctref(4,6), C16);
      _mm_store_ps(&Ctref(4,7), C17);
	    
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
    } else {
      if ( beta!=zero ) {
        L00 = _mm_load_ps(&Ccol(0,0));
        L01 = _mm_load_ps(&Ccol(0,1));
        L02 = _mm_load_ps(&Ccol(0,2));
        L03 = _mm_load_ps(&Ccol(0,3));
        C00 = beta*L00 + C00;
        C01 = beta*L01 + C01;
        C02 = beta*L02 + C02;
        C03 = beta*L03 + C03;
        L04 = _mm_load_ps(&Ccol(0,4));
        L05 = _mm_load_ps(&Ccol(0,5));
        L06 = _mm_load_ps(&Ccol(0,6));
        L07 = _mm_load_ps(&Ccol(0,7));
        C04 = beta*L04 + C04;
        C05 = beta*L05 + C05;
        C06 = beta*L06 + C06;
        C07 = beta*L07 + C07;
        
	L10 = _mm_load_ps(&Ccol(4,0));
        L11 = _mm_load_ps(&Ccol(4,1));
        L12 = _mm_load_ps(&Ccol(4,2));
        L13 = _mm_load_ps(&Ccol(4,3));
        C10 = beta*L10 + C10;
        C11 = beta*L11 + C11;
        C12 = beta*L12 + C12;
        C13 = beta*L13 + C13;
        L14 = _mm_load_ps(&Ccol(4,4));
        L15 = _mm_load_ps(&Ccol(4,5));
        L16 = _mm_load_ps(&Ccol(4,6));
        L17 = _mm_load_ps(&Ccol(4,7));
        C14 = beta*L14 + C14;
        C15 = beta*L15 + C15;
        C16 = beta*L16 + C16;
        C17 = beta*L17 + C17;
      }
	    
      _mm_store_ps(&Ccol(0,0), C00);
      _mm_store_ps(&Ccol(0,1), C01);
      _mm_store_ps(&Ccol(0,2), C02);
      _mm_store_ps(&Ccol(0,3), C03);
      _mm_store_ps(&Ccol(0,4), C04);
      _mm_store_ps(&Ccol(0,5), C05);
      _mm_store_ps(&Ccol(0,6), C06);
      _mm_store_ps(&Ccol(0,7), C07);
      
      _mm_store_ps(&Ccol(4,0), C10);
      _mm_store_ps(&Ccol(4,1), C11);
      _mm_store_ps(&Ccol(4,2), C12);
      _mm_store_ps(&Ccol(4,3), C13);
      _mm_store_ps(&Ccol(4,4), C14);
      _mm_store_ps(&Ccol(4,5), C15);
      _mm_store_ps(&Ccol(4,6), C16);
      _mm_store_ps(&Ccol(4,7), C17);
	    
    }

}

inline void gemm_microkernel_Cresident_neon_4x4_fp32( int mr, int nr, int kc, float alpha, float *Ar, float *Br, float beta, float *C, int ldC ){

    __m128   C00, C01, C02, C03, 
             A00, A01, A02, A03, 
             B0, A0 ; 

{

  int MR=4;
  int NR=4;
    int      i, j, k, baseA = 0, baseB = 0, ldCt = MR, Amr, Bnr;
    float  zero = 0.0, one = 1.0, *Aptr, *Bptr, Ctmp[MR*NR];
    
    if ( kc==0 ) return;

    C00 = _mm_set1_ps(0); 
    C01 = _mm_set1_ps(0); 
    C02 = _mm_set1_ps(0); 
    C03 = _mm_set1_ps(0); 

    Aptr = &Br[0];
    Bptr = &Ar[0];
    Amr  = NR;
    Bnr  = MR;
    
    if ( alpha!=zero ) {
	    for ( k=0; k<kc; k++ ) {
		    
		    A0 = _mm_load_ps(&Aptr[baseA]);
		    B0 = _mm_load_ps(&Bptr[baseB]);
		    
		    C00 += A0 * B0[0]; 
		    C01 += A0 * B0[1]; 
		    C02 += A0 * B0[2]; 
		    C03 += A0 * B0[3]; 
		    
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
	    _mm_store_ps(&Ctref(0,0), C00);
	    _mm_store_ps(&Ctref(0,1), C01);
	    _mm_store_ps(&Ctref(0,2), C02);
	    _mm_store_ps(&Ctref(0,3), C03);
	    
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
		    A00 = _mm_load_ps(&Ccol(0,0));
		    A01 = _mm_load_ps(&Ccol(0,1));
		    A02 = _mm_load_ps(&Ccol(0,2));
		    A03 = _mm_load_ps(&Ccol(0,3));
		    
		    C00 = beta*A00 + C00;
		    C01 = beta*A01 + C01;
		    C02 = beta*A02 + C02;
		    C03 = beta*A03 + C03;
		 
	    }
	    
	    _mm_store_ps(&Ccol(0,0), C00);
	    _mm_store_ps(&Ccol(0,1), C01);
	    _mm_store_ps(&Ccol(0,2), C02);
	    _mm_store_ps(&Ccol(0,3), C03);
	    
    }
    else {
	    printf("Error: Incorrect use of 4x4 micro-kernel with %d x %d block\n", mr, nr);
	    exit(-1);
    }
}
}



inline void ukernel_neon_4x4_np_fp32( int mr, int nr, int kc, float alpha, float *Ar, size_t ldA, float *Br, size_t ldB, float beta, float *C, int ldC ){

   __m128 C00, C01, C02, C03, C04, C05, C06, C07,
          C10, C11, C12, C13, C14, C15, C16, C17,
                A00, A01, A02, A03, 
                B00, B01, B02, B03, 
                A0, B0;
  const int MR=4;
  const int NR=4;
    int         i, j, k, baseA = 0, baseB = 0, ldCt = MR, Amr, Bnr;
  float  zero = 0.0, one = 1.0, *Aptr, *Bptr, Ctmp[MR*NR];
    
  if ( kc==0 ) return;
  C00 = _mm_set1_ps(0);
  C01 = _mm_set1_ps(0);
  C02 = _mm_set1_ps(0);
  C03 = _mm_set1_ps(0);
    
  Aptr = &Br[0];
  Bptr = &Ar[0];
  Amr  = NR;
  Bnr  = MR;
    
  if ( alpha!=zero ) {
    for ( k=0; k<kc; k++ ) {
       
      A0 = _mm_load_ps(&Aptr[baseA]);
      
      B00 = _mm_set1_ps(Bptr[baseB + 0]);
      B01 = _mm_set1_ps(Bptr[baseB + 1*ldA]);
      B02 = _mm_set1_ps(Bptr[baseB + 2*ldA]);
      B03 = _mm_set1_ps(Bptr[baseB + 3*ldA]);
      		    
      C00 += B00 * A0; 
      C01 += B01 * A0; 
      C02 += B02 * A0; 
      C03 += B03 * A0; 
      
      baseB += 1;
      baseA += ldB;
      
    }
  
    if ( alpha==-one ) {
      C00 = -C00; C01 = -C01; C02 = -C02; C03 = -C03; 
    } else if ( alpha!=one ) {
      C00 = alpha*C00; C01 = alpha*C01; C02 = alpha*C02; C03 = alpha*C03; 
    }

  }
    
  if ( (mr<MR)||(nr<NR) ) {
    _mm_store_ps(&Ctref(0,0), C00);
    _mm_store_ps(&Ctref(0,1), C01);
    _mm_store_ps(&Ctref(0,2), C02);
    _mm_store_ps(&Ctref(0,3), C03);
	    
    if ( beta!=zero ) {
      for ( j=0; j<nr; j++ ) 
        for ( i=0; i<mr; i++ ) 
	  Crow(i,j) = beta*Crow(i,j) + Ctrow(i,j);
    } else {
      for ( j=0; j<nr; j++ ) 
        for ( i=0; i<mr; i++ ) 
          Crow(i,j) = Ctrow(i,j);
    }
  } else if ( (mr==MR)&&(nr==NR) ) {
    if ( beta!=zero ) {
      A00 = _mm_load_ps(&Ccol(0,0));
      A01 = _mm_load_ps(&Ccol(0,1));
      A02 = _mm_load_ps(&Ccol(0,2));
      A03 = _mm_load_ps(&Ccol(0,3));
		    
      C00 = beta*A00 + C00;
      C01 = beta*A01 + C01;
      C02 = beta*A02 + C02;
      C03 = beta*A03 + C03;
   }
	    
   _mm_store_ps(&Ccol(0,0), C00);
   _mm_store_ps(&Ccol(0,1), C01);
   _mm_store_ps(&Ccol(0,2), C02);
   _mm_store_ps(&Ccol(0,3), C03);
	    
  } else {
    printf("Error: Incorrect use of 4x4 micro-kernel with %d x %d block\n", mr, nr);
    exit(-1);
  }
}



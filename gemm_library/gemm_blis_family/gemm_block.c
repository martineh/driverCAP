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

//#include <arm_neon.h>

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

void pack_RBA( int mc, int nc, float *M, int ldM, float *Mc, int RR );
void pack_CBA( int mc, int nc, float *M, int ldM, float *Mc, int RR );
void gemm_base_CresidentA( int m, int n, int k, 
                          float alpha, float *A, int ldA, 
                                       float *B, int ldB, 
                          float beta,  float *C, int ldC );

void gemm_block( size_t m, size_t n, size_t k, float alpha, float *A, size_t ldA, float *B, size_t ldB, float beta, float *C, size_t ldC, float *Ac, float *Bc, size_t MC, size_t NC, size_t KC ) {

  size_t    ic, jc, pc, mc, nc, kc, ir, jr, mr, nr; 
  float  zero = 0.0, one = 1.0, betaI; 
  float  *Aptr, *Bptr, *Cptr;

  // Quick return if possible
  if ( (m==0)||(n==0)||(((alpha==zero)||(k==0))&&(beta==one)) )
    return;

  //#include "quick_gemm.h"
  
  for ( jc=0; jc<n; jc+=NC ) {
    nc = min(n-jc, NC); 

    for ( pc=0; pc<k; pc+=KC ) {
      kc = min(k-pc, KC); 
      
      //if ( (transB=='N')&&(orderB=='C') )
        //Bptr = &Bcol(pc,jc);
      //else if ( (transB=='N')&&(orderB=='R') )
        Bptr = &Brow(pc,jc);
      //else if ( (transB=='T')&&(orderB=='C') )
        //Bptr = &Bcol(jc,pc);
      //else
        //Bptr = &Brow(jc,pc);
      
      pack_CBA( kc, nc, Bptr, ldB, Bc, NR);
      //pack_CB_v( orderB, transB, kc, nc, Bptr, ldB, Bc, NR);
      
      if ( pc==0 )
        betaI = beta;
      else
        betaI = one;
      
      for ( ic=0; ic<m; ic+=MC ) {
        mc = min(m-ic, MC); 
	
        //if ( (transA=='N')&&(orderA=='C') ){
          //Aptr = &Acol(ic, pc);
	//}else if ( (transA=='N')&&(orderA=='R') ){
          Aptr = &Arow(ic, pc);
	//}else if ( (transA=='T')&&(orderA=='C') ){
          //Aptr = &Acol(pc, ic);
	//}else{
          //Aptr = &Arow(pc, ic);
	//}
	
	//Comment or uncomment for packing or not
        pack_RBA( mc, kc, Aptr, ldA, Ac, MR);
        //pack_RB_v( orderA, transA, mc, kc, Aptr, ldA, Ac, MR);
	
	//#pragma omp  parallel for private(nr, ir, mr, Cptr) firstprivate (Ac, Bc)   
        for ( jr=0; jr<nc; jr+=NR ) {
          nr = min(nc-jr, NR); 
	  
          for ( ir=0; ir<mc; ir+=MR ) {
            mr = min(mc-ir, MR); 
	    
            //if ( orderC=='C' ) {
              //Cptr = &Ccol(ic+ir,jc+jr);
              //uKernels Stored By [Columns]
	      //gemm_microkernel_Cresident_AMD_avx256_2vx6_fp32( orderC, mr, nr, kc, alpha, &Ac[ir*kc], &Bc[jr*kc], betaI, Cptr, ldC ); 
	      //gemm_microkernel_Cresident_AMD_avx256_3vx4_fp32( orderC, mr, nr, kc, alpha, &Ac[ir*kc], &Bc[jr*kc], betaI, Cptr, ldC );
	    //} else {
              Cptr = &Crow(ic+ir,jc+jr);
	      //uKernels Stored By [Rows]
	      //--------------+--------------------------------------------------------------------------------------------------------
	      //No packing A  |
	      //--------------+--------------------------------------------------------------------------------------------------------
	      //gemm_microkernel_Cresident_AMD_avx256_6x2v_nopack_unroll_fp32( orderC, mr, nr, kc, alpha, &Arow(ic + ir, pc), ldA, &Bc[jr*kc], betaI, Cptr, ldC ); 
	      //gemm_microkernel_Cresident_AMD_avx256_6x2v_nopack_fp32( orderC, mr, nr, kc, alpha, &Arow(ic + ir, pc), ldA, &Bc[jr*kc], betaI, Cptr, ldC ); 
	      //gemm_microkernel_Cresident_AMD_avx256_4x3v_nopack_fp32( orderC, mr, nr, kc, alpha, &Arow(ic + ir, pc), ldA, &Bc[jr*kc], betaI, Cptr, ldC ); 
              
	      //--------------+--------------------------------------------------------------------------------------------------------
	      //Packing A     |
              //--------------+--------------------------------------------------------------------------------------------------------
	      //gemm_microkernel_Cresident_AMD_avx256_6x2v_fp32( orderC, mr, nr, kc, alpha, &Ac[ir*kc], &Bc[jr*kc], betaI, Cptr, ldC ); 
	      //gemm_microkernel_Cresident_AMD_avx256_4x3v_fp32( orderC, mr, nr, kc, alpha, &Ac[ir*kc], &Bc[jr*kc], betaI, Cptr, ldC ); 
	      //gemm_kernel(mr, nr, kc, &alpha, &Ac[ir*kc], &Bc[jr*kc], &betaI,  Cptr, 1, ldC, aux, cntx);
	      //bli_sgemm_haswell_asm_6x16(mr, nr, kc, &alpha, &Ac[ir*kc], &Bc[jr*kc], &betaI,  Cptr, 1, ldC, aux, cntx);
	    //}

            //#if defined(BASE)
	       gemm_base_CresidentA( mr, nr, kc, alpha, &Ac[ir*kc], MR, &Bc[jr*kc], NR, betaI, Cptr, ldC );
           //#elif defined(MK_BLIS)
               //gemm_kernel(mr, nr, kc, &alpha, &Ac[ir*kc], &Bc[jr*kc], betaI, Cptr, 1, ldC, aux, cntx);
            //#else
              //if(MR==mr && NR==nr ) {
                //gemm_microkernel_Cresident_AMD_avx256_mrxnr_fp32( orderC, mr, nr, kc, alpha, &Ac[ir*kc], &Bc[jr*kc], betaI, Cptr, ldC );
              //} else {
		//--------------+--------------------------------------------------------------------------------------------------------
                //MR=8 x NR=n   |
		//--------------+
	        //gemm_microkernel_Cresident_AMD_avx256_8x8_fp32( orderC, mr, nr, kc, alpha, &Ac[ir*kc], &Bc[jr*kc], betaI, Cptr, ldC );
	        //gemm_microkernel_Cresident_AMD_avx256_8x14_fp32( orderC, mr, nr, kc, alpha, &Ac[ir*kc], &Bc[jr*kc], betaI, Cptr, ldC ); 
		//-----------------------------------------------------------------------------------------------------------------------
		
		//--------------+--------------------------------------------------------------------------------------------------------
                //MR=16 x NR=n  |
		//--------------+
	        //gemm_microkernel_Cresident_AMD_avx256_16x6_optimum_fp32( orderC, mr, nr, kc, alpha, &Ac[ir*kc], &Bc[jr*kc], betaI, Cptr, ldC ); 
	        //gemm_microkernel_Cresident_AMD_avx256_16x6_BLIS_fp32( orderC, mr, nr, kc, alpha, &Ac[ir*kc], &Bc[jr*kc], betaI, Cptr, ldC ); 
	        //gemm_microkernel_Cresident_AMD_avx256_6x16_BLIS_ROW_fp32( orderC, mr, nr, kc, alpha, &Ac[ir*kc], &Bc[jr*kc], betaI, Cptr, ldC ); 
		//-----------------------------------------------------------------------------------------------------------------------
		
		//--------------+--------------------------------------------------------------------------------------------------------
                //MR=24 x NR=n  |
		//--------------+
	       // gemm_microkernel_Cresident_AMD_avx256_24x4_fp32( orderC, mr, nr, kc, alpha, &Ac[ir*kc], &Bc[jr*kc], betaI, Cptr, ldC );
		//-----------------------------------------------------------------------------------------------------------------------
		
		//--------------+--------------------------------------------------------------------------------------------------------
                // MR=32 x NR=n |
		//--------------+--------------------------------------------------------------------------------------------------------
	        //gemm_microkernel_Cresident_AMD_avx256_32x2_fp32( orderC, mr, nr, kc, alpha, &Ac[ir*kc], &Bc[jr*kc], betaI, Cptr, ldC );
		//-----------------------------------------------------------------------------------------------------------------------
		
		//--------------+--------------------------------------------------------------------------------------------------------
                // MR=40 x NR=n |
		//--------------+--------------------------------------------------------------------------------------------------------
	        //gemm_microkernel_Cresident_AMD_avx256_40x2_fp32( orderC, mr, nr, kc, alpha, &Ac[ir*kc], &Bc[jr*kc], betaI, Cptr, ldC );
		//-----------------------------------------------------------------------------------------------------------------------
              //}
            //#endif
          }
	  
        }
      }
    }
  }
}


void pack_RBA( int mc, int nc, float *M, int ldM, float *Mc, int RR ){
/*
  BLIS pack for M-->Mc
*/
  int    i, j, ii, k, rr;

  /*if ( ((transM=='N')&&( orderM=='C'))||
       ((transM=='T')&&( orderM=='R')) ) {
    for ( i=0; i<mc; i+=RR ) { 
      k = i*nc;
      rr = min( mc-i, RR );
      
      for ( j=0; j<nc; j++ ) {
        for ( ii=0; ii<rr; ii++ ) {
	  Mc[k] = Mcol(i+ii, j);
          k++;
        } 
        k += (RR-rr);
      }
    }
  } else {*/
    //printf("Pack RB, by row\n");
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
  //}
}

void pack_CBA( int mc, int nc, float *M, int ldM, float *Mc, int RR ){
/*
  BLIS pack for M-->Mc
*/
  int    i, j, jj, k, nr;

  /*k = 0;
  if ( ((transM=='N')&&( orderM=='C'))||
       ((transM=='T')&&( orderM=='R')) )
    for ( j=0; j<nc; j+=RR ) { 
      k = j*mc;
      nr = min( nc-j, RR );
      for ( i=0; i<mc; i++ ) {
        for ( jj=0; jj<nr; jj++ ) {
          Mc[k] = Mcol(i,j+jj);
          k++;
        }
        k += (RR-nr);
      }
    }
  else*/
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

void gemm_base_CresidentA( int m, int n, int k, 
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


/* 
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

   author    = "Héctor Martínez Pérez" contact   = "el2mapeh@uco.es"
   copyright = "Copyright 2023, Universidad de Córdoba"
   license   = "GPLv3"
   status    = "Production"
   version   = "1.1"
*/

#include "gemm_library.h"
#include <omp.h>

#define Mcol(a1,a2)  M[ (a2)*(ldM)+(a1) ]
#define min(a,b) (((a)<(b))?(a):(b))

void pack_allA( int MM, int KK, int MC, int KC, float *M, int ldM, float *Mc, int RR ){
  int    i, j, ii, k, rr;
  int mc, kc;

  for (int mm = 0; mm < MM; mm += MC) {
    mc = min(MM - mm, MC);
    for (int kk = 0; kk < KK; kk += KC) {
      kc = min(KK - kk, KC);
      for ( i=0; i<mc; i+=RR ) {
        k = i*kc;
        rr = min( mc-i, RR );
        for ( j=0; j<kc; j++ ) {
          for ( ii=0; ii<rr; ii++ ) {
             Mc[k] = Mcol(j,i+ii);
            k++;
          }
          k += (RR-rr);
        }
      }
    }
  }

}


double gemm_reference(DT* A, DT* B, DT* C, size_t m, size_t n, size_t k, int mode, double *error) {
  
  DT *Ac, *Bc, *Ct, *Cloop;

  unsigned int nreps;
  double tmult, t1, t2, flops, gflops;
  float tmin   = 1.0;
  //I don't know this config
  size_t MC, NC, KC;
  //Optimus config for Carmel Jetson Xavier

  if (mode == 3) {
    MC = 32;
    NC = 32;
    KC = 128;
  } else {
    MC = 512;
    NC = 800;
    KC = 1024;
  }

  float alpha = 1.0;
  float beta  = 1.0;

  Cloop = (DT *) calloc ((m + m % 4) * (n + n % 4), sizeof(DT));
  memset(C,  0, (m + (m % 4))* (n + n % 4) * sizeof(DT));

  Ac = (DT *) calloc (m * k * 2, sizeof(DT)); 
  Bc = (DT *) calloc (k * n * 2, sizeof(DT)); 
  
  packA(Ac, A, m, k); 
  packB(Bc, B, k, n); 
  pack_allA( m, k, MC, KC, A, k, Ac, 8 );
   
  //===========================================//
  //   YOUR MATRIX MULTIPLICATION EVALUATION   // 
  //===========================================//
    omp_set_dynamic(0);
    omp_set_num_threads(1);

  nreps = 1;
  t1 = dclock();
  if (mode == 1) //EASY
    _saxpy_gemm2(A, B, C, m, n, k);
  else if (mode == 2) //MEDIU
    _saxpy_gemm(A, B, C, m, n, k);
  else if (mode == 3) //HARD
    gemm_block_ukernel_max( m, n, k, alpha, A, k, B, n, beta, C, n, Ac, Bc, MC, NC, KC);
  else if (mode == 4) //VERY HARD
    gemm_block_ukernel_max( m, n, k, alpha, A, k, B, n, beta, C, n, Ac, Bc, MC, NC, KC);
    //gemm_block_ukernel_max( m, n, k, alpha, A, k, B, n, beta, C, n, Ac, Bc, MC, NC, KC);
    //gemm_block_ukernel_8x8( m, n, k, alpha, Ac, k, B, n, beta, C, n, Ac, Bc, MC, NC, KC);

  t2    = dclock();
  tmult = ( t2 > t1 ? t2 - t1 : 0.0 );

  
  if (tmult < tmin) {
    nreps = 0;
    t1 = dclock();
    while (tmult <= tmin) {
      nreps++;
      if (mode == 1)
        _saxpy_gemm2(A, B, Cloop, m, n, k);
      else if (mode == 2)
        _saxpy_gemm(A, B, Cloop, m, n, k);
      else if (mode == 3)
        gemm_block_ukernel_max( m, n, k, alpha, A, k, B, n, beta, Cloop, n, Ac, Bc, MC, NC, KC);
      else if (mode == 4) //VERY HARD
        gemm_block_ukernel_max( m, n, k, alpha, A, k, B, n, beta, Cloop, n, Ac, Bc, MC, NC, KC);
       // gemm_block_ukernel_max( m, n, k, alpha, A, k, B, n, beta, Cloop, n, Ac, Bc, MC, NC, KC);
      
      t2    = dclock();
      tmult = t2 - t1;
    }
  }

  tmult = tmult / nreps;
  flops  = 2.0 * m * n * k;
  gflops = flops / (1.0e+9 * tmult );
  //===========================================//
  
  /** RESULT VALIDATION: Test Matrix **/
  Ct = (DT *) calloc ((m + m % 4) * (n + n % 4), sizeof(DT));
  base_matrix_multiplication(A, B, Ct, m, n, k);
  *error = validate_multiplication(C, Ct, m, n);
  free(Ct);
    
  //Only for packing
  free(Ac);
  free(Bc);
  free(Cloop);

  return gflops;

}


//-------------------------------------------------
// MATRIX MULTIPLICATIONS DEFINITIONS
//-------------------------------------------------
void naive_gemm(DT *A, DT *B, DT *C, size_t M, size_t N, size_t K) {
  size_t i, j, k;
  
  //Leading Dimensions
  size_t ldA, ldB, ldC;
  DT Atmp;

  ldA = K;
  ldB = N;
  ldC = N;

  for (i = 0; i < M; i++) {
    for (k = 0; k < K; k++) {
      Atmp = Arow(i, k);
      for (j = 0; j < N; j++) {
        Crow(i, j) += Atmp * Brow(k, j);
      }
    }
  }

}

void _saxpy_gemm(float *A, float *B, float *C, size_t M, size_t N, size_t K) {
    //----------------------------------------------------
    // x0  x4  x8      y0  y4  y8     x0y0 + x4y1 + x8y2
    // x1  x5  x9      y1  y5  y9     x1y0 + x5y1 + x9y2
    // x2  x6  xA   X  y2  y6  yA  =  x2y0 + x6y1 + xAy2
    // x3  x7  xB      y3  y7  yB     x3y0 + x7y1 + xBy2
    //----------------------------------------------------
    size_t m, n, k;
    float saxpy;
    //Leading Dimensions
    size_t ldA, ldB, ldC;

    ldA = K;
    ldB = N;
    ldC = N;

    for (m = 0; m < M; m++)
    {
        float *c = &Crow(m, 0);
        for (k = 0; k < K; k++)
        {
            float *b = &Brow(k, 0);
            float tmp = Arow(m, k);
            for (n = 0; n < N; n++)
            {
                c[n] += tmp * b[n];
            }
        }
    }

}

void _saxpy_gemm2(float *A, float *B, float *C, size_t M, size_t N, size_t K) {
    //----------------------------------------------------
    // x0  x4  x8      y0  y4  y8     x0y0 + x4y1 + x8y2
    // x1  x5  x9      y1  y5  y9     x1y0 + x5y1 + x9y2
    // x2  x6  xA   X  y2  y6  yA  =  x2y0 + x6y1 + xAy2
    // x3  x7  xB      y3  y7  yB     x3y0 + x7y1 + xBy2
    //----------------------------------------------------
    size_t m, n, k;
    float saxpy;
    //Leading Dimensions
    size_t ldA, ldB, ldC;

    ldA = K;
    ldB = N;
    ldC = N;

    int K_new = (K + (4 - K % 4));

    float *Apad = (float*)calloc((M + (4 - M % 4))*K_new, sizeof(float));

    float *Bpad = (float*)calloc((N + (4 - N % 4))*K_new, sizeof(float));

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            Apad[i*K_new + j] = Arow(i, j);
        }
    }
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            Bpad[j*K_new + i] = Brow(i, j);
        }
    }
    // matrix multiplication with SIMD 128
    __m128 *rowA, *rowB_T;
    __m128 p, s;
    for (int i = 0; i < M; ++i) {
        float * res_tmp = &Crow(i, 0);
        for (int j = 0; j < N; ++j) {
            rowA = (__m128 *) &Apad[i*K_new];
            rowB_T = (__m128 *) &Bpad[j*K_new];
            s = _mm_setzero_ps();
            for (int k = 0; k < K_new / 4; ++k) {
                p = _mm_mul_ps(rowA[k], rowB_T[k]);
                s = _mm_add_ps(s, p);
            }
            p = _mm_movehl_ps(p,s);
            s = _mm_add_ps(s,p);
            p = _mm_shuffle_ps(s,s,1);
            s = _mm_add_ss(s,p);
            _mm_store_ss(&res_tmp[j], s);
        }
    }
}

void naive_unroll_gemm(DT *A, DT *B, DT *C, size_t M, size_t N, size_t K) {
  size_t i, j, k;
  size_t n_left;
  
  const unsigned char vlen = 4;
  
  //Leading Dimensions
  size_t ldA, ldB, ldC;
  DT Atmp;
  
  n_left  = N % vlen;

  ldA = K;
  ldB = N;
  ldC = N;

  for (i = 0; i < M; i++) {
    for (k = 0; k < K; k++) {
      Atmp = Arow(i, k);
      
      for (j = 0; j < n_left; j++) {
        Crow(i, j) += Atmp * Brow(k, j);
      }
      
      for (j = n_left; j < N; j += 4) {
        Crow(i, j + 0) += Atmp * Brow(k, j + 0);
        Crow(i, j + 1) += Atmp * Brow(k, j + 1);
        Crow(i, j + 2) += Atmp * Brow(k, j + 2);
        Crow(i, j + 3) += Atmp * Brow(k, j + 3);
      }
     }
   }
}

//INSTRINSICS ONLY FOR FLOAT32
void naive_block_intrinsics_gemm(DT *A, DT *B, DT *C, size_t M, size_t N, size_t K) {
  size_t i, j, k;
  
  __m128 _A, _B, _C;  
  
  //Leading Dimensions
  size_t ldA, ldB, ldC;
  //size_t k_left = K % A_BLOCK;
  
  //k_left = K;

  ldA = K;
  ldB = N;
  ldC = N;

  for (i = 0; i < M; i++) {
    for (k = 0; k < K; k++) { //A_BLOCK
      _A = _mm_set_ps1(Arow(i, k));
      for (j = 0; j < N; j += B_BLOCK) {
        _C = _mm_load_ps(&Crow(i, j));
        _B = _mm_load_ps(&Brow(k,   j));
	_C += _A * _B;
      }
    }
  }

}

//INSTRINSICS ONLY FOR FLOAT32
void naive_intrinsics_gemm(DT *A, DT *B, DT *C, size_t M, size_t N, size_t K) {
  size_t i, j, k;
  
  __m128 _A, _B, _C;  
  
  //Leading Dimensions
  size_t ldA, ldB, ldC;
  
  ldA = K;
  ldB = N;
  ldC = N;

  for (i = 0; i < M; i++) {
   for (k = 0; k < K; k++) { //A_BLOCK
      _A = _mm_set_ps1(Arow(i, k));
      for (j = 0; j < N; j += B_BLOCK) {
        _C = _mm_load_ps(&Crow(i, j));
        _B = _mm_load_ps(&Brow(k, j));
	_C += _A * _B;
     }
   }
 }

}

void packA(DT *Ac, DT *A, size_t m, size_t k) {
  size_t ldAc = k + (k % 4);
  for (size_t i = 0; i < m; i++)
    for (size_t j = 0; j < k; j++)
      Ac[i * ldAc + j] = A[i * k + j];
}

void packACol(DT *Ac, DT *A, size_t m, size_t k) {
  size_t ldAc = m + (m % 4);
  for (size_t j = 0; j < k; j++)
    for (size_t i = 0; i < m; i++)
      Ac[j * ldAc + i] = A[i * k + j];
}

void packB(DT *Bc, DT *B, size_t k, size_t n) {
  size_t ldBc = n + (n % 4);
  for (size_t j = 0; j < k; j++)
    for (size_t h = 0; h < n; h++)
      Bc[j * ldBc + h] = B[j * n + h];
  
}

void base_matrix_multiplication(DT *A, DT *B, DT *C, size_t m, size_t n, size_t k) {
  size_t h, i, j;
  //Leading Dimensions
  size_t ldA, ldB, ldC;
  
  ldA = k;
  ldB = n;
  ldC = n;

  for (h = 0; h < n; h++) {
    for (i = 0; i < m; i++) {
      for (j = 0; j < k; j++) {
        Crow(i, h) += Arow(i, j) * Brow(j, h);
      }
    }
  }

}

void saxpy_matrix_multiplication(DT *A, DT *B, DT *C, size_t m, size_t n, size_t k) {
  //----------------------------------------------------
  // x0  x4  x8      y0  y4  y8     x0y0 + x4y1 + x8y2
  // x1  x5  x9      y1  y5  y9     x1y0 + x5y1 + x9y2
  // x2  x6  xA   X  y2  y6  yA  =  x2y0 + x6y1 + xAy2
  // x3  x7  xB      y3  y7  yB     x3y0 + x7y1 + xBy2
  //----------------------------------------------------
  size_t h, i, j;
  DT saxpy; 
  //Leading Dimensions
  size_t ldA, ldB, ldC;
  
  ldA = k;
  ldB = n;
  ldC = n;

  for (h = 0; h < n; h++) {
    for (j = 0; j < k; j++) {
      saxpy = Brow(j, h);
      for (i = 0; i < m; i++) {
        Crow(i, h) += Arow(i, j) * saxpy;
      }
    }
  }
}

//Validate Function. DON'T CHANGE THIS FUNCTION!
double validate_multiplication(DT *C, DT *Ct, size_t m, size_t n) {

  double error = 0.0;
  double nrm   = 0.0;
  double tmp;
  
  size_t ldC  = n;
  size_t ldCt = n;
  
  for ( size_t i = 0; i < m; i++ ) {
    for ( size_t j = 0; j < n; j++ ) {
      tmp = (double) Ctrow(i, j);
      nrm += tmp*tmp;
      tmp = (double) dabs(Crow(i, j) - Ctrow(i, j));
      error += tmp*tmp;
    }
  }

  if ( nrm != 0.0 )
    error = sqrt(error) / sqrt(nrm);
  else
    error = sqrt(error);

  return error;

}

//-------------------------------------------------
//  TIMERS DEFINITIONS
//-------------------------------------------------
double dclock() {
  //Timer
  struct timeval  tv;
  gettimeofday( &tv, NULL );
  return (double) (tv.tv_sec + tv.tv_usec*1.0e-6);
}


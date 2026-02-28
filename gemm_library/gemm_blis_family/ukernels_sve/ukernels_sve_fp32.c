/**
 * This file is part of convDirect
 *
 * Copyright (C) 2021-22 Universitat Politècnica de València and
 *                       Universitat Jaume I
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#include "ukernels_sve_fp32.h"

#define  Acol(a1, a2)    A[ (a2) * (ldA)  + (a1) ]
#define  Bcol(a1, a2)    B[ (a2) * (ldB)  + (a1) ]
#define  Ccol(a1, a2)    C[ (a2) * (ldC)  + (a1) ]
#define Ctcol(a1, a2) Ctmp[ (a2) * (ldCt) + (a1) ]

#define  Arow(a1, a2)    A[ (a1) * (ldA)  + (a2) ]
#define  Brow(a1, a2)    B[ (a1) * (ldB)  + (a2) ]
#define  Crow(a1, a2)    C[ (a1) * (ldC)  + (a2) ]
#define Ctrow(a1, a2) Ctmp[ (a1) * (ldCt) + (a2) ]

#define Ctref(a1, a2) Ctmp[ (a2) * (ldCt) + (a1) ]
#define Atref(a1, a2) Atmp[ (a2) * (Atlda)+ (a1) ]



#define Atrow(a1, a2) Atmp[ (a1)*(ldAt)+(a2) ]
#define Btrow(a1, a2) Btmp[ (a1)*(ldBt)+(a2) ]
#define Ctref(a1, a2) Ctmp[ (a2)*(ldCt)+(a1) ]
#define Atref(a1, a2) Atmp[ (a2)*(ldAt)+(a1) ]


//#define SET_MR_NR(_MR, _NR)			\
//    const int MR = _MR;     \
//    const int NR = _NR;


//=============================================================================================//
//====================== 32 X 10    A S M    M I C R O - K E R N E L ===========================//
//=============================================================================================//
#include <stdint.h>

#define LABEL(str) "   ." #str": \n\t"
#define BEQ(str) "b.eq ." #str"  \n\t"
#define BNE(str) "b.ne ." #str"  \n\t"
#define BRANCH(str) "b ." #str"  \n\t"


void gemm_microkernel_sve_asm_32x10_fp32( int kc, float *Ar, float *Br, float *C, int ldC, float alpha, float beta ) {
/*
  BLIS GEMM microkernel, computes the product Cr := Cr + Ar * Br
  Specific: only for MRxNR = 10x12
*/

  if (kc == 0)
    return;
  
  uint64_t ukc           = kc / 4;
  uint64_t ukc_left      = kc % 4;
  uint64_t uldC          = ldC;
  const uint64_t whilelo = 16;

  //printf("kc=%d, ukc=%ld, ukc_left=%ld\n", kc, ukc, ukc_left);
  
  __asm__ volatile
  (
   // input operands
  " ldr x20, %[Baddr]                          \n\t" // A
  " ldr x21, %[Aaddr]                          \n\t" // B
  " ldr x22, %[uldC]                           \n\t" // Load ldC
  " lsl x22, x22, #2                           \n\t" // Stride multiplied by 4 (sizeof(FP32))
  " ldr x23, %[ukc]                            \n\t" 
  " ldr x24, %[ukc_left]                       \n\t" 
  " ldr x25, %[whilelo]                        \n\t"
  "                                            \n\t"
  " mov x0, xzr                                \n\t" //Preparing whilelelo SVE (16)
  " whilelo p0.s, x0, x25                      \n\t"
  "                                            \n\t"
  " ld1w  z28.s, p0/z, [x21]                   \n\t" // Load B
  " ld1w  z29.s, p0/z, [x21, #1, mul vl]       \n\t" // Load B + 16
  "                                            \n\t"
  " add x21, x21, #128                         \n\t" // Update B Address = (NR(32) *FP32(4))
  "                                            \n\t"
  " ld1rw z20.s, p0/z, [x20]                   \n\t" // Load and Broadcast A 1/10
  " ld1rw z21.s, p0/z, [x20, 4]                \n\t" // Load and Broadcast A 2/10
  " ld1rw z22.s, p0/z, [x20, 8]                \n\t" // Load and Broadcast A 3/10
  " ld1rw z23.s, p0/z, [x20, 12]               \n\t" // Load and Broadcast A 4/10
  " ld1rw z24.s, p0/z, [x20, 16]               \n\t" // Load and Broadcast A 5/10
  " ld1rw z25.s, p0/z, [x20, 20]               \n\t" // Load and Broadcast A 6/10
  " ld1rw z26.s, p0/z, [x20, 24]               \n\t" // Load and Broadcast A 7/10
  " ld1rw z27.s, p0/z, [x20, 28]               \n\t" // Load and Broadcast A 8/10
  "                                            \n\t" 
  " ldr   x0, %[Caddr]                         \n\t" //Load C Address
  " add   x1, x0, x22                          \n\t" 
  " add   x2, x1, x22                          \n\t" 
  " add   x3, x2, x22                          \n\t" 
  " add   x4, x3, x22                          \n\t" 
  " add   x5, x4, x22                          \n\t" 
  " add   x6, x5, x22                          \n\t" 
  " add   x7, x6, x22                          \n\t" 
  " add   x8, x7, x22                          \n\t" 
  " add   x9, x8, x22                          \n\t"   
  "                                            \n\t"
  " prfm  PLDL1STRM, [x0]                      \n\t" // Prefetch C  
  " prfm  PLDL1STRM, [x1]                      \n\t"
  " prfm  PLDL1STRM, [x2]                      \n\t"
  " prfm  PLDL1STRM, [x3]                      \n\t"
  " prfm  PLDL1STRM, [x4]                      \n\t"
  " prfm  PLDL1STRM, [x5]                      \n\t"
  " prfm  PLDL1STRM, [x6]                      \n\t"
  " prfm  PLDL1STRM, [x7]                      \n\t"
  " prfm  PLDL1STRM, [x8]                      \n\t"
  " prfm  PLDL1STRM, [x9]                      \n\t"  
  "                                            \n\t"
  " dup z0.s,  #0                              \n\t" //Clear C
  " dup z1.s,  #0                              \n\t"
  " dup z2.s,  #0                              \n\t"
  " dup z3.s,  #0                              \n\t"
  " dup z4.s,  #0                              \n\t"
  " dup z5.s,  #0                              \n\t"
  " dup z6.s,  #0                              \n\t"
  " dup z7.s,  #0                              \n\t"
  " dup z8.s,  #0                              \n\t"
  " dup z9.s,  #0                              \n\t"
  " dup z10.s, #0                              \n\t"
  " dup z11.s, #0                              \n\t"
  " dup z12.s, #0                              \n\t"
  " dup z13.s, #0                              \n\t"
  " dup z14.s, #0                              \n\t"
  " dup z15.s, #0                              \n\t"
  " dup z16.s, #0                              \n\t"
  " dup z17.s, #0                              \n\t"
  " dup z18.s, #0                              \n\t"
  " dup z19.s, #0                              \n\t"
  "                                            \n\t"
  " cmp x23, 0                                 \n\t" // Check end of iteration count   
  BEQ(LEFT_SVE_10x32)
  LABEL(LOOP_SVE_10x32)                            
  "                                            \n\t" //------------------------------
  "                                            \n\t" // [*] LOOP UNROLL x1
  "                                            \n\t" //------------------------------
  "                                            \n\t"
  " ld1w  z30.s, p0/z, [x21]                   \n\t" // Load and Broadcast B
  " ld1w  z31.s, p0/z, [x21, #1, mul vl]       \n\t" // Load and Broadcast B + 16
  "                                            \n\t"
  " fmla z0.s, p0/m, z28.s, z20.s              \n\t"
  " fmla z1.s, p0/m, z29.s, z20.s              \n\t"
  " ld1rw z20.s, p0/z, [x20, 32]               \n\t" // Load and Broadcast A 9/10
  "                                            \n\t"
  " fmla z2.s, p0/m, z28.s, z21.s              \n\t"
  " fmla z3.s, p0/m, z29.s, z21.s              \n\t"
  " ld1rw z21.s, p0/z, [x20, 36]               \n\t" // Load and Broadcast A 10/10
  "                                            \n\t"
  " fmla z4.s, p0/m, z28.s, z22.s              \n\t"
  " fmla z5.s, p0/m, z29.s, z22.s              \n\t"
  " ld1rw z22.s, p0/z, [x20, 40]               \n\t" // Load and Broadcast A 1/10
  "                                            \n\t"
  " fmla z6.s, p0/m, z28.s, z23.s              \n\t"
  " fmla z7.s, p0/m, z29.s, z23.s              \n\t"
  " ld1rw z23.s, p0/z, [x20, 44]               \n\t" // Load and Broadcast A 2/10
  "                                            \n\t"
  " fmla z8.s, p0/m, z28.s, z24.s              \n\t"
  " fmla z9.s, p0/m, z29.s, z24.s              \n\t"
  " ld1rw z24.s, p0/z, [x20, 48]               \n\t" // Load and Broadcast A 3/10
  "                                            \n\t"
  " fmla z10.s, p0/m, z28.s, z25.s             \n\t"
  " fmla z11.s, p0/m, z29.s, z25.s             \n\t"
  " ld1rw z25.s, p0/z, [x20, 52]               \n\t" // Load and Broadcast A 4/10
  "                                            \n\t"
  " fmla z12.s, p0/m, z28.s, z26.s             \n\t"
  " fmla z13.s, p0/m, z29.s, z26.s             \n\t"
  " ld1rw z26.s, p0/z, [x20, 56]               \n\t" // Load and Broadcast A 5/10
  "                                            \n\t"
  " fmla z14.s, p0/m, z28.s, z27.s             \n\t"
  " fmla z15.s, p0/m, z29.s, z27.s             \n\t"
  " ld1rw z27.s, p0/z, [x20, 60]               \n\t" // Load and Broadcast A 6/10
  "                                            \n\t"
  " fmla z16.s, p0/m, z28.s, z20.s             \n\t"
  " fmla z17.s, p0/m, z29.s, z20.s             \n\t"
  " ld1rw z20.s, p0/z, [x20, 64]               \n\t" // Load and Broadcast A 7/10
  "                                            \n\t"
  " fmla z18.s, p0/m, z28.s, z21.s             \n\t"
  " fmla z19.s, p0/m, z29.s, z21.s             \n\t"
  " ld1rw z21.s, p0/z, [x20, 68]               \n\t" // Load and Broadcast A 8/10
  "                                            \n\t"
  " add x21, x21, #128                         \n\t" // Update B Address = (NR(32) * FP32(4))
  "                                            \n\t" 
  "                                            \n\t" //------------------------------
  "                                            \n\t" // [*] LOOP UNROLL x2
  "                                            \n\t" //------------------------------
  "                                            \n\t"
  " ld1w  z28.s, p0/z, [x21]                   \n\t" // Load B
  " ld1w  z29.s, p0/z, [x21, #1, mul vl]       \n\t" // Load B + 16
  "                                            \n\t" 
  " fmla z0.s, p0/m, z30.s, z22.s              \n\t"
  " fmla z1.s, p0/m, z31.s, z22.s              \n\t"
  " ld1rw z22.s, p0/z, [x20, 72]               \n\t" // Load and Broadcast A 9/10
  "                                            \n\t"
  " fmla z2.s, p0/m, z30.s, z23.s              \n\t"
  " fmla z3.s, p0/m, z31.s, z23.s              \n\t"
  " ld1rw z23.s, p0/z, [x20, 76]               \n\t" // Load and Broadcast A 10/10
  "                                            \n\t"
  " fmla z4.s, p0/m, z30.s, z24.s              \n\t"
  " fmla z5.s, p0/m, z31.s, z24.s              \n\t"
  " ld1rw z24.s, p0/z, [x20, 80]               \n\t" // Load and Broadcast A 1/10
  "                                            \n\t"
  " fmla z6.s, p0/m, z30.s, z25.s              \n\t"
  " fmla z7.s, p0/m, z31.s, z25.s              \n\t"
  " ld1rw z25.s, p0/z, [x20, 84]               \n\t" // Load and Broadcast A 2/10
  "                                            \n\t"
  " fmla z8.s, p0/m, z30.s, z26.s              \n\t"
  " fmla z9.s, p0/m, z31.s, z26.s              \n\t"
  " ld1rw z26.s, p0/z, [x20, 88]               \n\t" // Load and Broadcast A 3/10
  "                                            \n\t"
  " fmla z10.s, p0/m, z30.s, z27.s             \n\t"
  " fmla z11.s, p0/m, z31.s, z27.s             \n\t"
  " ld1rw z27.s, p0/z, [x20, 92]               \n\t" // Load and Broadcast A 4/10
  "                                            \n\t"
  " fmla z12.s, p0/m, z30.s, z20.s             \n\t"
  " fmla z13.s, p0/m, z31.s, z20.s             \n\t"
  " ld1rw z20.s, p0/z, [x20, 96]               \n\t" // Load and Broadcast A 5/10
  "                                            \n\t"
  " fmla z14.s, p0/m, z30.s, z21.s             \n\t"
  " fmla z15.s, p0/m, z31.s, z21.s             \n\t"
  " ld1rw z21.s, p0/z, [x20, 100]              \n\t" // Load and Broadcast A 6/10
  "                                            \n\t"
  " fmla z16.s, p0/m, z30.s, z22.s             \n\t"
  " fmla z17.s, p0/m, z31.s, z22.s             \n\t"
  " ld1rw z22.s, p0/z, [x20, 104]              \n\t" // Load and Broadcast A 7/10
  "                                            \n\t"
  " fmla z18.s, p0/m, z30.s, z23.s             \n\t"
  " fmla z19.s, p0/m, z31.s, z23.s             \n\t"
  " ld1rw z23.s, p0/z, [x20, 108]              \n\t" // Load and Broadcast A 8/10
  "                                            \n\t"
  " add x21, x21, #128                         \n\t" // Update B Address = (NR(32) *FP32(4))
  "                                            \n\t"
  "                                            \n\t" //------------------------------
  "                                            \n\t" // [*] LOOP UNROLL x3
  "                                            \n\t" //------------------------------
  "                                            \n\t"
  " ld1w  z30.s, p0/z, [x21]                   \n\t" // Load and Broadcast B
  " ld1w  z31.s, p0/z, [x21, #1, mul vl]       \n\t" // Load and Broadcast B + 16
  "                                            \n\t"
  " fmla z0.s, p0/m, z28.s, z24.s              \n\t"
  " fmla z1.s, p0/m, z29.s, z24.s              \n\t"
  " ld1rw z24.s, p0/z, [x20, 112]              \n\t" // Load and Broadcast A 9/10
  "                                            \n\t"
  " fmla z2.s, p0/m, z28.s, z25.s              \n\t"
  " fmla z3.s, p0/m, z29.s, z25.s              \n\t"
  " ld1rw z25.s, p0/z, [x20, 116]              \n\t" // Load and Broadcast A 10/10
  "                                            \n\t"
  " fmla z4.s, p0/m, z28.s, z26.s              \n\t"
  " fmla z5.s, p0/m, z29.s, z26.s              \n\t"
  " ld1rw z26.s, p0/z, [x20, 120]              \n\t" // Load and Broadcast A 1/10
  "                                            \n\t"
  " fmla z6.s, p0/m, z28.s, z27.s              \n\t"
  " fmla z7.s, p0/m, z29.s, z27.s              \n\t"
  " ld1rw z27.s, p0/z, [x20, 124]              \n\t" // Load and Broadcast A 2/10
  "                                            \n\t"
  " fmla z8.s, p0/m, z28.s, z20.s              \n\t"
  " fmla z9.s, p0/m, z29.s, z20.s              \n\t"
  " ld1rw z20.s, p0/z, [x20, 128]              \n\t" // Load and Broadcast A 3/10
  "                                            \n\t"
  " fmla z10.s, p0/m, z28.s, z21.s             \n\t"
  " fmla z11.s, p0/m, z29.s, z21.s             \n\t"
  " ld1rw z21.s, p0/z, [x20, 132]              \n\t" // Load and Broadcast A 4/10
  "                                            \n\t"
  " fmla z12.s, p0/m, z28.s, z22.s             \n\t"
  " fmla z13.s, p0/m, z29.s, z22.s             \n\t"
  " ld1rw z22.s, p0/z, [x20, 136]              \n\t" // Load and Broadcast A 5/10
  "                                            \n\t"
  " fmla z14.s, p0/m, z28.s, z23.s             \n\t"
  " fmla z15.s, p0/m, z29.s, z23.s             \n\t"
  " ld1rw z23.s, p0/z, [x20, 140]              \n\t" // Load and Broadcast A 6/10
  "                                            \n\t"
  " fmla z16.s, p0/m, z28.s, z24.s             \n\t"
  " fmla z17.s, p0/m, z29.s, z24.s             \n\t"
  " ld1rw z24.s, p0/z, [x20, 144]              \n\t" // Load and Broadcast A 7/10
  "                                            \n\t"
  " fmla z18.s, p0/m, z28.s, z25.s             \n\t"
  " fmla z19.s, p0/m, z29.s, z25.s             \n\t"
  " ld1rw z25.s, p0/z, [x20, 148]              \n\t" // Load and Broadcast A 8/10
  "                                            \n\t"
  " add x21, x21, #128                         \n\t" // Update B Address = (NR(32) *FP32(4))
  "                                            \n\t" 
  "                                            \n\t" //------------------------------
  "                                            \n\t" // [*] LOOP UNROLL x4
  "                                            \n\t" //------------------------------
  "                                            \n\t"
  " ld1w  z28.s, p0/z, [x21]                   \n\t" // Load B
  " ld1w  z29.s, p0/z, [x21, #1, mul vl]       \n\t" // Load B + 16
  "                                            \n\t"
  " fmla z0.s, p0/m, z30.s, z26.s              \n\t"
  " fmla z1.s, p0/m, z31.s, z26.s              \n\t"
  " ld1rw z26.s, p0/z, [x20, 152]              \n\t" // Load and Broadcast A 9/10
  "                                            \n\t"
  " fmla z2.s, p0/m, z30.s, z27.s              \n\t"
  " fmla z3.s, p0/m, z31.s, z27.s              \n\t"
  " ld1rw z27.s, p0/z, [x20, 156]              \n\t" // Load and Broadcast A 10/10
  "                                            \n\t"
  " add x20, x20, #160                         \n\t" // Update A Address = (MR(10) *FP32(4))
  "                                            \n\t"
  " fmla z4.s, p0/m, z30.s, z20.s              \n\t"
  " fmla z5.s, p0/m, z31.s, z20.s              \n\t"
  " ld1rw z20.s, p0/z, [x20]                   \n\t" // Load and Broadcast A 1/10
  "                                            \n\t"
  " fmla z6.s, p0/m, z30.s, z21.s              \n\t"
  " fmla z7.s, p0/m, z31.s, z21.s              \n\t"
  " ld1rw z21.s, p0/z, [x20, 4]                \n\t" // Load and Broadcast A 9/10
  "                                            \n\t"
  " fmla z8.s, p0/m, z30.s, z22.s              \n\t"
  " fmla z9.s, p0/m, z31.s, z22.s              \n\t"
  " ld1rw z22.s, p0/z, [x20, 8]                \n\t" // Load and Broadcast A 9/10
  "                                            \n\t"
  " fmla z10.s, p0/m, z30.s, z23.s             \n\t"
  " fmla z11.s, p0/m, z31.s, z23.s             \n\t"
  " ld1rw z23.s, p0/z, [x20, 12]               \n\t" // Load and Broadcast A 9/10
  "                                            \n\t"
  " fmla z12.s, p0/m, z30.s, z24.s             \n\t"
  " fmla z13.s, p0/m, z31.s, z24.s             \n\t"
  " ld1rw z24.s, p0/z, [x20, 16]               \n\t" // Load and Broadcast A 9/10
  "                                            \n\t"
  " fmla z14.s, p0/m, z30.s, z25.s             \n\t"
  " fmla z15.s, p0/m, z31.s, z25.s             \n\t"
  " ld1rw z25.s, p0/z, [x20, 20]               \n\t" // Load and Broadcast A 9/10
  "                                            \n\t"
  " fmla z16.s, p0/m, z30.s, z26.s             \n\t"
  " fmla z17.s, p0/m, z31.s, z26.s             \n\t"
  " ld1rw z26.s, p0/z, [x20, 24]               \n\t" // Load and Broadcast A 9/10
  "                                            \n\t"
  " fmla z18.s, p0/m, z30.s, z27.s             \n\t"
  " fmla z19.s, p0/m, z31.s, z27.s             \n\t"
  " ld1rw z27.s, p0/z, [x20, 28]               \n\t" // Load and Broadcast A 9/10
  "                                            \n\t"
  " add x21, x21, #128                         \n\t" // Update B Address = (NR(32) *FP32(4))
  "                                            \n\t"
  " sub x23, x23, 1                            \n\t" // Decrease iteration count by 1
  " cmp x23, 0                                 \n\t" // Check end of iteration count   
  BNE(LOOP_SVE_10x32)
  LABEL(LEFT_SVE_10x32)
  "                                            \n\t"
  " cmp x24, 0                                 \n\t" // Check end of iteration count   
  BEQ(SVE_BETA)
  "                                            \n\t" //------------------------------
  "                                            \n\t" // [*] K LEFT, (x1)
  "                                            \n\t" //------------------------------
  " fmla z0.s, p0/m, z28.s, z20.s              \n\t"
  " fmla z1.s, p0/m, z29.s, z20.s              \n\t"
  " ld1rw z20.s, p0/z, [x20, 32]               \n\t"
  "                                            \n\t"
  " fmla z2.s, p0/m, z28.s, z21.s              \n\t"
  " fmla z3.s, p0/m, z29.s, z21.s              \n\t"
  " ld1rw z21.s, p0/z, [x20, 36]               \n\t"
  "                                            \n\t"
  " fmla z4.s, p0/m, z28.s, z22.s              \n\t"
  " fmla z5.s, p0/m, z29.s, z22.s              \n\t"
  "                                            \n\t"
  " fmla z6.s, p0/m, z28.s, z23.s              \n\t"
  " fmla z7.s, p0/m, z29.s, z23.s              \n\t"
  "                                            \n\t"
  " fmla z8.s, p0/m, z28.s, z24.s              \n\t"
  " fmla z9.s, p0/m, z29.s, z24.s              \n\t"
  "                                            \n\t"
  " fmla z10.s, p0/m, z28.s, z25.s             \n\t"
  " fmla z11.s, p0/m, z29.s, z25.s             \n\t"
  "                                            \n\t"
  " fmla z12.s, p0/m, z28.s, z26.s             \n\t"
  " fmla z13.s, p0/m, z29.s, z26.s             \n\t"
  "                                            \n\t"
  " fmla z14.s, p0/m, z28.s, z27.s             \n\t"
  " fmla z15.s, p0/m, z29.s, z27.s             \n\t"
  "                                            \n\t"
  " fmla z16.s, p0/m, z28.s, z20.s             \n\t"
  " fmla z17.s, p0/m, z29.s, z20.s             \n\t"
  "                                            \n\t"
  " fmla z18.s, p0/m, z28.s, z21.s             \n\t"
  " fmla z19.s, p0/m, z29.s, z21.s             \n\t"
  "                                            \n\t"
  " sub x24, x24, 1                            \n\t"
  " cmp x24, 0                                 \n\t" 
  BEQ(SVE_BETA)
  "                                            \n\t" //------------------------------
  "                                            \n\t" // [*] K LEFT, (x2)
  "                                            \n\t" //------------------------------
  " add x20, x20, #40                          \n\t" // Update A Address = (MR(10) * FP32(4))
  //" add x21, x21, #128                         \n\t" // Update B Address = (NR(32) * FP32(4))  
  "                                            \n\t"
  " ld1w  z28.s, p0/z, [x21]                   \n\t" 
  " ld1w  z29.s, p0/z, [x21, #1, mul vl]       \n\t" 
  "                                            \n\t"
  " ld1rw z20.s, p0/z, [x20]                   \n\t" 
  " ld1rw z21.s, p0/z, [x20, 4]                \n\t" 
  " ld1rw z22.s, p0/z, [x20, 8]                \n\t" 
  " ld1rw z23.s, p0/z, [x20, 12]               \n\t" 
  " ld1rw z24.s, p0/z, [x20, 16]               \n\t" 
  " ld1rw z25.s, p0/z, [x20, 20]               \n\t" 
  " ld1rw z26.s, p0/z, [x20, 24]               \n\t" 
  " ld1rw z27.s, p0/z, [x20, 28]               \n\t" 
  "                                            \n\t" 
  " fmla z0.s, p0/m, z28.s, z20.s              \n\t"
  " fmla z1.s, p0/m, z29.s, z20.s              \n\t"
  " ld1rw z20.s, p0/z, [x20, 32]               \n\t"
  "                                            \n\t"
  " fmla z2.s, p0/m, z28.s, z21.s              \n\t"
  " fmla z3.s, p0/m, z29.s, z21.s              \n\t"
  " ld1rw z21.s, p0/z, [x20, 36]               \n\t"
  "                                            \n\t"
  " fmla z4.s, p0/m, z28.s, z22.s              \n\t"
  " fmla z5.s, p0/m, z29.s, z22.s              \n\t"
  "                                            \n\t"
  " fmla z6.s, p0/m, z28.s, z23.s              \n\t"
  " fmla z7.s, p0/m, z29.s, z23.s              \n\t"
  "                                            \n\t"
  " fmla z8.s, p0/m, z28.s, z24.s              \n\t"
  " fmla z9.s, p0/m, z29.s, z24.s              \n\t"
  "                                            \n\t"
  " fmla z10.s, p0/m, z28.s, z25.s             \n\t"
  " fmla z11.s, p0/m, z29.s, z25.s             \n\t"
  "                                            \n\t"
  " fmla z12.s, p0/m, z28.s, z26.s             \n\t"
  " fmla z13.s, p0/m, z29.s, z26.s             \n\t"
  "                                            \n\t"
  " fmla z14.s, p0/m, z28.s, z27.s             \n\t"
  " fmla z15.s, p0/m, z29.s, z27.s             \n\t"
  "                                            \n\t"
  " fmla z16.s, p0/m, z28.s, z20.s             \n\t"
  " fmla z17.s, p0/m, z29.s, z20.s             \n\t"
  "                                            \n\t"
  " fmla z18.s, p0/m, z28.s, z21.s             \n\t"
  " fmla z19.s, p0/m, z29.s, z21.s             \n\t"
  "                                            \n\t"
  " sub x24, x24, 1                            \n\t"
  " cmp x24, 0                                 \n\t"
  BEQ(SVE_BETA)
  "                                            \n\t" //------------------------------
  "                                            \n\t" // [*] K LEFT, (x3) <<Last>>
  "                                            \n\t" //------------------------------
  " add x20, x20, #40                          \n\t" // Update A Address = (MR(10) * FP32(4))
  " add x21, x21, #128                         \n\t" // Update B Address = (NR(32) * FP32(4))  
  "                                            \n\t"
  " ld1w  z28.s, p0/z, [x21]                   \n\t" 
  " ld1w  z29.s, p0/z, [x21, #1, mul vl]       \n\t" 
  "                                            \n\t"
  " ld1rw z20.s, p0/z, [x20]                   \n\t" 
  " ld1rw z21.s, p0/z, [x20, 4]                \n\t" 
  " ld1rw z22.s, p0/z, [x20, 8]                \n\t" 
  " ld1rw z23.s, p0/z, [x20, 12]               \n\t" 
  " ld1rw z24.s, p0/z, [x20, 16]               \n\t" 
  " ld1rw z25.s, p0/z, [x20, 20]               \n\t" 
  " ld1rw z26.s, p0/z, [x20, 24]               \n\t" 
  " ld1rw z27.s, p0/z, [x20, 28]               \n\t" 
  "                                            \n\t" 
  " fmla z0.s, p0/m, z28.s, z20.s              \n\t"
  " fmla z1.s, p0/m, z29.s, z20.s              \n\t"
  " ld1rw z20.s, p0/z, [x20, 32]               \n\t"
  "                                            \n\t"
  " fmla z2.s, p0/m, z28.s, z21.s              \n\t"
  " fmla z3.s, p0/m, z29.s, z21.s              \n\t"
  " ld1rw z21.s, p0/z, [x20, 36]               \n\t"
  "                                            \n\t"
  " fmla z4.s, p0/m, z28.s, z22.s              \n\t"
  " fmla z5.s, p0/m, z29.s, z22.s              \n\t"
  "                                            \n\t"
  " fmla z6.s, p0/m, z28.s, z23.s              \n\t"
  " fmla z7.s, p0/m, z29.s, z23.s              \n\t"
  "                                            \n\t"
  " fmla z8.s, p0/m, z28.s, z24.s              \n\t"
  " fmla z9.s, p0/m, z29.s, z24.s              \n\t"
  "                                            \n\t"
  " fmla z10.s, p0/m, z28.s, z25.s             \n\t"
  " fmla z11.s, p0/m, z29.s, z25.s             \n\t"
  "                                            \n\t"
  " fmla z12.s, p0/m, z28.s, z26.s             \n\t"
  " fmla z13.s, p0/m, z29.s, z26.s             \n\t"
  "                                            \n\t"
  " fmla z14.s, p0/m, z28.s, z27.s             \n\t"
  " fmla z15.s, p0/m, z29.s, z27.s             \n\t"
  "                                            \n\t"
  " fmla z16.s, p0/m, z28.s, z20.s             \n\t"
  " fmla z17.s, p0/m, z29.s, z20.s             \n\t"
  "                                            \n\t"
  " fmla z18.s, p0/m, z28.s, z21.s             \n\t"
  " fmla z19.s, p0/m, z29.s, z21.s             \n\t"  
  "                                            \n\t"
  LABEL(SVE_BETA)
  " ldr  x27, %[beta]                          \n\t" //Beta
  " dup  z30.s, w27                            \n\t"
  " fcmp s30, #0.0                             \n\t"
  BEQ(SVE_STORE)
  "                                            \n\t"
  //------------------------------------------------
  "                                            \n\t"
  " ld1w  z20.s, p0/z,  [x0]                   \n\t" // Load C
  " ld1w  z21.s, p0/z,  [x0, #1, mul vl]       \n\t" // 
  " ld1w  z22.s, p0/z,  [x1]                   \n\t"
  " ld1w  z23.s, p0/z,  [x1, #1, mul vl]       \n\t" // 
  " ld1w  z24.s, p0/z,  [x2]                   \n\t"
  " ld1w  z25.s, p0/z,  [x2, #1, mul vl]       \n\t" // 
  " ld1w  z26.s, p0/z,  [x3]                   \n\t"
  " ld1w  z27.s, p0/z,  [x3, #1, mul vl]       \n\t" // 
  " ld1w  z28.s, p0/z,  [x4]                   \n\t"
  " ld1w  z29.s, p0/z,  [x4, #1, mul vl]       \n\t" //
  "                                            \n\t"
  " fmla z0.s, p0/m, z30.s, z20.s              \n\t"
  " fmla z1.s, p0/m, z30.s, z21.s              \n\t"
  "                                            \n\t"
  " fmla z2.s, p0/m, z30.s, z22.s              \n\t"
  " fmla z3.s, p0/m, z30.s, z23.s              \n\t"
  "                                            \n\t"
  " fmla z4.s, p0/m, z30.s, z24.s              \n\t"
  " fmla z5.s, p0/m, z30.s, z25.s              \n\t"
  "                                            \n\t"
  " fmla z6.s, p0/m, z30.s, z26.s              \n\t"
  " fmla z7.s, p0/m, z30.s, z27.s              \n\t"
  "                                            \n\t"
  " fmla z8.s, p0/m, z30.s, z28.s              \n\t"
  " fmla z9.s, p0/m, z30.s, z29.s              \n\t"
  "                                            \n\t"
  " ld1w  z20.s, p0/z,  [x5]                   \n\t"
  " ld1w  z21.s, p0/z,  [x5, #1, mul vl]       \n\t"
  " ld1w  z22.s, p0/z,  [x6]                   \n\t"
  " ld1w  z23.s, p0/z,  [x6, #1, mul vl]       \n\t"
  " ld1w  z24.s, p0/z,  [x7]                   \n\t"
  " ld1w  z25.s, p0/z,  [x7, #1, mul vl]       \n\t"
  " ld1w  z26.s, p0/z,  [x8]                   \n\t"
  " ld1w  z27.s, p0/z,  [x8, #1, mul vl]       \n\t" // Load C + 16
  " ld1w  z28.s, p0/z,  [x9]                   \n\t"
  " ld1w  z29.s, p0/z,  [x9, #1, mul vl]       \n\t" 
  "                                            \n\t"
  " fmla z10.s, p0/m, z30.s, z20.s             \n\t"
  " fmla z11.s, p0/m, z30.s, z21.s             \n\t"
  "                                            \n\t"
  " fmla z12.s, p0/m, z30.s, z22.s             \n\t"
  " fmla z13.s, p0/m, z30.s, z23.s             \n\t"
  "                                            \n\t"
  " fmla z14.s, p0/m, z30.s, z24.s             \n\t"
  " fmla z15.s, p0/m, z30.s, z25.s             \n\t"
  "                                            \n\t"
  " fmla z16.s, p0/m, z30.s, z26.s             \n\t"
  " fmla z17.s, p0/m, z30.s, z27.s             \n\t"
  "                                            \n\t"
  " fmla z18.s, p0/m, z30.s, z28.s             \n\t"
  " fmla z19.s, p0/m, z30.s, z29.s             \n\t"    
  "                                            \n\t"
  //------------------------------------------------
  "                                            \n\t"
  LABEL(SVE_STORE)
  "                                            \n\t"
  " st1w z0.s,  p0, [x0]                       \n\t" // Store C x0
  " st1w z1.s,  p0, [x0, #1, mul vl]           \n\t" // Store C x0 + 16
  " st1w z2.s,  p0, [x1]                       \n\t" 
  " st1w z3.s,  p0, [x1, #1, mul vl]           \n\t" 
  " st1w z4.s,  p0, [x2]                       \n\t" 
  " st1w z5.s,  p0, [x2, #1, mul vl]           \n\t" 
  " st1w z6.s,  p0, [x3]                       \n\t" 
  " st1w z7.s,  p0, [x3, #1, mul vl]           \n\t" 
  " st1w z8.s,  p0, [x4]                       \n\t" 
  " st1w z9.s,  p0, [x4, #1, mul vl]           \n\t" 
  " st1w z10.s, p0, [x5]                       \n\t" 
  " st1w z11.s, p0, [x5, #1, mul vl]           \n\t" 
  " st1w z12.s, p0, [x6]                       \n\t" 
  " st1w z13.s, p0, [x6, #1, mul vl]           \n\t" 
  " st1w z14.s, p0, [x7]                       \n\t" 
  " st1w z15.s, p0, [x7, #1, mul vl]           \n\t" 
  " st1w z16.s, p0, [x8]                       \n\t" 
  " st1w z17.s, p0, [x8, #1, mul vl]           \n\t" 
  " st1w z18.s, p0, [x9]                       \n\t" 
  " st1w z19.s, p0, [x9, #1, mul vl]           \n\t" 
  "                                            \n\t"
  LABEL(FIN)
  : // output operands (none)
  : // input operands
    [ukc]      "m" (ukc),     
    [ukc_left] "m" (ukc_left),
    [Aaddr]    "m" (Ar),      
    [Baddr]    "m" (Br),      
    [Caddr]    "m" (C),
    [uldC]     "m" (uldC),     
    [whilelo]  "m" (whilelo),
    [alpha]    "m" (alpha),
    [beta]     "m" (beta)
  : // Memory Dir. Register 
    "x0",   "x1",  "x2",  "x3",  "x4", 
    "x5",   "x6",  "x7",  "x8",  "x9",
    "x20", "x21", "x22", "x23", "x24",
    "x25", "x26", "x27",
    // SVE Register 
    "z0",  "z1",  "z2",  "z3",
    "z4",  "z5",  "z6",  "z7",
    "z8",  "z9", "z10", "z11",
    "z12", "z13", "z14", "z15",
    "z16", "z17", "z18", "z19",
    "z20", "z21", "z22", "z23",
    "z24", "z25", "z26", "z27",
    "z28", "z29", "z30", "z31");
  
}




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



inline void gemm_microkernel_sve_32x10_fp32( char orderC, int mr, int nr, int kc, float alpha, float *Ar, float *Br, float beta, float *C, int ldC ) {

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

  svbool_t pred16 = svwhilelt_b32_u32(0, pred);

  Aptr = &Ar[0];
  Bptr = &Br[0];
  Amr  = MR;
  Bnr  = NR;  

  //printf("@mr=%d, nr=%d\n", mr, nr);
  
  if (nr <= 4) {
    if (mr <= 16) {
      
      svbool_t predmr = svwhilelt_b32_u32(0, mr);
      //----------------------------------------------
      B0 = svdup_n_f32(Bptr[baseB + 0]); 
      B1 = svdup_n_f32(Bptr[baseB + 1]); 
      B2 = svdup_n_f32(Bptr[baseB + 2]); 
      B3 = svdup_n_f32(Bptr[baseB + 3]); 
      //----------------------------------------------
      
      //----------------------------------------------
      A0 = svld1_f32(predmr, &Aptr[baseA]);
      //----------------------------------------------
    
      //----------------------------------------------
      //Prefetching [C]
      //----------------------------------------------    
      svprfw(predmr, &Ccol(0, 0), SV_PLDL1STRM);
      svprfw(predmr, &Ccol(0, 1), SV_PLDL1STRM);
      svprfw(predmr, &Ccol(0, 2), SV_PLDL1STRM);
      svprfw(predmr, &Ccol(0, 3), SV_PLDL1STRM);      
      //----------------------------------------------
    
      //------------------------
      //Clear columns
      //------------------------
      C00 = svdup_n_f32(0); 
      C01 = svdup_n_f32(0); 
      C02 = svdup_n_f32(0); 
      C03 = svdup_n_f32(0); 
      //-------------------------
      
      for ( k=0; k < kc; k++ ) {
	
	baseB = baseB + NR;
	
	C00 = svmla_f32_z(predmr, C00, A0, B0);
	B0  = svdup_n_f32(Bptr[baseB + 0]); 
      	
	C01 = svmla_f32_z(predmr, C01, A0, B1);
	B1  = svdup_n_f32(Bptr[baseB + 1]); 
	
	C02 = svmla_f32_z(predmr, C02, A0, B2);
	B2  = svdup_n_f32(Bptr[baseB + 2]); 
	
	C03 = svmla_f32_z(predmr, C03, A0, B3);
	B3  = svdup_n_f32(Bptr[baseB + 3]); 
	
	baseA = baseA + MR;
	
	A0 = svld1_f32(predmr, &Aptr[baseA]);
	
      }
      
      svst1_f32(predmr, &Ctref(0, 0),  C00);
      svst1_f32(predmr, &Ctref(0, 1),  C01);
      svst1_f32(predmr, &Ctref(0, 2),  C02);
      svst1_f32(predmr, &Ctref(0, 3),  C03);
      
    } else {
      
      //----------------------------------------------
      B0 = svdup_n_f32(Bptr[baseB + 0]); 
      B1 = svdup_n_f32(Bptr[baseB + 1]); 
      B2 = svdup_n_f32(Bptr[baseB + 2]); 
      B3 = svdup_n_f32(Bptr[baseB + 3]); 
      //----------------------------------------------
      
      //----------------------------------------------
      A0 = svld1_f32(pred16, &Aptr[baseA]);
      A1 = svld1_f32(pred16, &Aptr[baseA + 16]);
      //----------------------------------------------
      
      //----------------------------------------------
      //Prefetching [C]
      //----------------------------------------------    
      svprfw(pred16, &Ccol(0, 0), SV_PLDL1STRM);
      svprfw(pred16, &Ccol(0, 1), SV_PLDL1STRM);
      svprfw(pred16, &Ccol(0, 2), SV_PLDL1STRM);
      svprfw(pred16, &Ccol(0, 3), SV_PLDL1STRM);
      
      svprfw(pred16, &Ccol(16, 0), SV_PLDL1STRM);
      svprfw(pred16, &Ccol(16, 1), SV_PLDL1STRM);
      svprfw(pred16, &Ccol(16, 2), SV_PLDL1STRM);
      svprfw(pred16, &Ccol(16, 3), SV_PLDL1STRM);
      //----------------------------------------------
      
      //------------------------
      //Clear columns
      //------------------------
      C00 = svdup_n_f32(0); 
      C01 = svdup_n_f32(0); 
      C02 = svdup_n_f32(0); 
      C03 = svdup_n_f32(0); 
      
      C10 = svdup_n_f32(0); 
      C11 = svdup_n_f32(0); 
      C12 = svdup_n_f32(0); 
      C13 = svdup_n_f32(0); 
      //-------------------------
      
      for ( k=0; k < kc; k++ ) {
	
	baseB = baseB + NR;
	
	C00 = svmla_f32_z(pred16, C00, A0, B0);
	C10 = svmla_f32_z(pred16, C10, A1, B0);
	B0  = svdup_n_f32(Bptr[baseB + 0]); 
      	
	C01 = svmla_f32_z(pred16, C01, A0, B1);
	C11 = svmla_f32_z(pred16, C11, A1, B1);
	B1  = svdup_n_f32(Bptr[baseB + 1]); 
	
	C02 = svmla_f32_z(pred16, C02, A0, B2);
	C12 = svmla_f32_z(pred16, C12, A1, B2);
	B2  = svdup_n_f32(Bptr[baseB + 2]); 
	
	C03 = svmla_f32_z(pred16, C03, A0, B3);
	C13 = svmla_f32_z(pred16, C13, A1, B3);
	B3  = svdup_n_f32(Bptr[baseB + 3]); 
	
	baseA = baseA + MR;
	
	A0 = svld1_f32(pred16, &Aptr[baseA]);
	A1 = svld1_f32(pred16, &Aptr[baseA + 16]);
	
      }
      
      svst1_f32(pred16, &Ctref(0, 0),  C00);
      svst1_f32(pred16, &Ctref(0, 1),  C01);
      svst1_f32(pred16, &Ctref(0, 2),  C02);
      svst1_f32(pred16, &Ctref(0, 3),  C03);
      
      svst1_f32(pred16, &Ctref(16, 0),  C10);
      svst1_f32(pred16, &Ctref(16, 1),  C11);
      svst1_f32(pred16, &Ctref(16, 2),  C12);
      svst1_f32(pred16, &Ctref(16, 3),  C13);
    }
  } else if (nr <= 6) {
    if (mr <= 16) {
      
      svbool_t predmr = svwhilelt_b32_u32(0, mr);
      //----------------------------------------------
      B0 = svdup_n_f32(Bptr[baseB + 0]); 
      B1 = svdup_n_f32(Bptr[baseB + 1]); 
      //----------------------------------------------
      
      //----------------------------------------------
      A0 = svld1_f32(predmr, &Aptr[baseA]);
      //----------------------------------------------
      
      //----------------------------------------------
      //Prefetching [C]
      //----------------------------------------------    
      svprfw(predmr, &Ccol(0, 0), SV_PLDL1STRM);
      svprfw(predmr, &Ccol(0, 1), SV_PLDL1STRM);
      svprfw(predmr, &Ccol(0, 2), SV_PLDL1STRM);
      svprfw(predmr, &Ccol(0, 3), SV_PLDL1STRM);
      svprfw(predmr, &Ccol(0, 4), SV_PLDL1STRM);
      svprfw(predmr, &Ccol(0, 5), SV_PLDL1STRM);      
      //----------------------------------------------
      
      //------------------------
      //Clear columns
      //------------------------
      C00 = svdup_n_f32(0); 
      C01 = svdup_n_f32(0); 
      C02 = svdup_n_f32(0); 
      C03 = svdup_n_f32(0); 
      C04 = svdup_n_f32(0); 
      C05 = svdup_n_f32(0); 
      //-------------------------
    
      for ( k=0; k < kc; k++ ) {      
	
	C00 = svmla_f32_z(predmr, C00, A0, B0);
	B0  = svdup_n_f32(Bptr[baseB + 2]); 
      	
	C01 = svmla_f32_z(predmr, C01, A0, B1);
	B1  = svdup_n_f32(Bptr[baseB + 3]);      
	
	C02 = svmla_f32_z(predmr, C02, A0, B0);
	B0  = svdup_n_f32(Bptr[baseB + 4]); 
	
	C03 = svmla_f32_z(predmr, C03, A0, B1);
	B1  = svdup_n_f32(Bptr[baseB + 5]); 
	
	baseB = baseB + NR;
	C04 = svmla_f32_z(predmr, C04, A0, B0);
	B0  = svdup_n_f32(Bptr[baseB + 0]); 
	
	C05 = svmla_f32_z(predmr, C05, A0, B1);
	B1  = svdup_n_f32(Bptr[baseB + 1]); 
	
	
	baseA = baseA + MR;
	
	A0 = svld1_f32(predmr, &Aptr[baseA]);
	
      }

      svst1_f32(predmr, &Ctref(0, 0),  C00);
      svst1_f32(predmr, &Ctref(0, 1),  C01);
      svst1_f32(predmr, &Ctref(0, 2),  C02);
      svst1_f32(predmr, &Ctref(0, 3),  C03);
      svst1_f32(predmr, &Ctref(0, 4),  C04);
      svst1_f32(predmr, &Ctref(0, 5),  C05);
      
    } else {
      //----------------------------------------------
      B0 = svdup_n_f32(Bptr[baseB + 0]); 
      B1 = svdup_n_f32(Bptr[baseB + 1]); 
      //----------------------------------------------
      
      //----------------------------------------------
      A0 = svld1_f32(pred16, &Aptr[baseA]);
      A1 = svld1_f32(pred16, &Aptr[baseA + 16]);
      //----------------------------------------------
    
      //----------------------------------------------
      //Prefetching [C]
      //----------------------------------------------    
      svprfw(pred16, &Ccol(0, 0), SV_PLDL1STRM);
      svprfw(pred16, &Ccol(0, 1), SV_PLDL1STRM);
      svprfw(pred16, &Ccol(0, 2), SV_PLDL1STRM);
      svprfw(pred16, &Ccol(0, 3), SV_PLDL1STRM);
      svprfw(pred16, &Ccol(0, 4), SV_PLDL1STRM);
      svprfw(pred16, &Ccol(0, 5), SV_PLDL1STRM);
      
      svprfw(pred16, &Ccol(16, 0), SV_PLDL1STRM);
      svprfw(pred16, &Ccol(16, 1), SV_PLDL1STRM);
      svprfw(pred16, &Ccol(16, 2), SV_PLDL1STRM);
      svprfw(pred16, &Ccol(16, 3), SV_PLDL1STRM);
      svprfw(pred16, &Ccol(16, 4), SV_PLDL1STRM);
      svprfw(pred16, &Ccol(16, 5), SV_PLDL1STRM);
      //----------------------------------------------
      
      //------------------------
      //Clear columns
      //------------------------
      C00 = svdup_n_f32(0); 
      C01 = svdup_n_f32(0); 
      C02 = svdup_n_f32(0); 
      C03 = svdup_n_f32(0); 
      C04 = svdup_n_f32(0); 
      C05 = svdup_n_f32(0); 
      
      C10 = svdup_n_f32(0); 
      C11 = svdup_n_f32(0); 
      C12 = svdup_n_f32(0); 
      C13 = svdup_n_f32(0);
      C14 = svdup_n_f32(0); 
      C15 = svdup_n_f32(0); 
      //-------------------------
      
      for ( k=0; k < kc; k++ ) {      
	
	C00 = svmla_f32_z(pred16, C00, A0, B0);
	C10 = svmla_f32_z(pred16, C10, A1, B0);
	B0  = svdup_n_f32(Bptr[baseB + 2]); 
      	
	C01 = svmla_f32_z(pred16, C01, A0, B1);
	C11 = svmla_f32_z(pred16, C11, A1, B1);
	B1  = svdup_n_f32(Bptr[baseB + 3]);      
	
	C02 = svmla_f32_z(pred16, C02, A0, B0);
	C12 = svmla_f32_z(pred16, C12, A1, B0);
	B0  = svdup_n_f32(Bptr[baseB + 4]); 
	
	C03 = svmla_f32_z(pred16, C03, A0, B1);
	C13 = svmla_f32_z(pred16, C13, A1, B1);
	B1  = svdup_n_f32(Bptr[baseB + 5]); 
	
	baseB = baseB + NR;
	C04 = svmla_f32_z(pred16, C04, A0, B0);
	C14 = svmla_f32_z(pred16, C14, A1, B0);
	B0  = svdup_n_f32(Bptr[baseB + 0]); 
	
	C05 = svmla_f32_z(pred16, C05, A0, B1);
	C15 = svmla_f32_z(pred16, C15, A1, B1);
	B1  = svdup_n_f32(Bptr[baseB + 1]); 
	
	
	baseA = baseA + MR;
	
	A0 = svld1_f32(pred16, &Aptr[baseA]);
	A1 = svld1_f32(pred16, &Aptr[baseA + 16]);
	
      }
      
      svst1_f32(pred16, &Ctref(0, 0),  C00);
      svst1_f32(pred16, &Ctref(0, 1),  C01);
      svst1_f32(pred16, &Ctref(0, 2),  C02);
      svst1_f32(pred16, &Ctref(0, 3),  C03);
      svst1_f32(pred16, &Ctref(0, 4),  C04);
      svst1_f32(pred16, &Ctref(0, 5),  C05);
      
      svst1_f32(pred16, &Ctref(16, 0),  C10);
      svst1_f32(pred16, &Ctref(16, 1),  C11);
      svst1_f32(pred16, &Ctref(16, 2),  C12);
      svst1_f32(pred16, &Ctref(16, 3),  C13);
      svst1_f32(pred16, &Ctref(16, 4),  C14);
      svst1_f32(pred16, &Ctref(16, 5),  C15);
    }
  } else if (nr <= 8) {  // 6 < nr <= 8 
    if (mr <= 16) {
      
      svbool_t predmr = svwhilelt_b32_u32(0, mr);

      //----------------------------------------------
      B0 = svdup_n_f32(Bptr[baseB + 0]); 
      B1 = svdup_n_f32(Bptr[baseB + 1]);
      B2 = svdup_n_f32(Bptr[baseB + 2]); 
      B3 = svdup_n_f32(Bptr[baseB + 3]);
      //----------------------------------------------
      
      //----------------------------------------------
      A0 = svld1_f32(predmr, &Aptr[baseA]);
      baseA = baseA + MR;
      //----------------------------------------------
      
      //----------------------------------------------
      //Prefetching [C]
      //----------------------------------------------    
      svprfw(predmr, &Ccol(0, 0), SV_PLDL1STRM);
      svprfw(predmr, &Ccol(0, 1), SV_PLDL1STRM);
      svprfw(predmr, &Ccol(0, 2), SV_PLDL1STRM);
      svprfw(predmr, &Ccol(0, 3), SV_PLDL1STRM);
      svprfw(predmr, &Ccol(0, 4), SV_PLDL1STRM);
      svprfw(predmr, &Ccol(0, 5), SV_PLDL1STRM);
      svprfw(predmr, &Ccol(0, 6), SV_PLDL1STRM);
      svprfw(predmr, &Ccol(0, 7), SV_PLDL1STRM);
      //----------------------------------------------
      
      //------------------------
      //Clear columns
      //------------------------
      C00 = svdup_n_f32(0); 
      C01 = svdup_n_f32(0); 
      C02 = svdup_n_f32(0); 
      C03 = svdup_n_f32(0); 
      C04 = svdup_n_f32(0); 
      C05 = svdup_n_f32(0); 
      C06 = svdup_n_f32(0); 
      C07 = svdup_n_f32(0); 
      //-------------------------
      
      for ( k=0; k < kc; k ++) {
	
	C00 = svmla_f32_z(predmr, C00, A0, B0);
	B0  = svdup_n_f32(Bptr[baseB + 4]); 
      
	C01 = svmla_f32_z(predmr, C01, A0, B1);
	B1  = svdup_n_f32(Bptr[baseB + 5]);      
      
	C02 = svmla_f32_z(predmr, C02, A0, B2);
	B2  = svdup_n_f32(Bptr[baseB + 6]); 
      
	C03 = svmla_f32_z(predmr, C03, A0, B3);
	B3  = svdup_n_f32(Bptr[baseB + 7]); 

	baseB = baseB + NR;
	
	C04 = svmla_f32_z(predmr, C04, A0, B0);
	B0  = svdup_n_f32(Bptr[baseB + 0]); 
	
	C05 = svmla_f32_z(predmr, C05, A0, B1);
	B1  = svdup_n_f32(Bptr[baseB + 1]); 
	
	C06 = svmla_f32_z(predmr, C06, A0, B2);
	B2  = svdup_n_f32(Bptr[baseB + 2]); 
      
	C07 = svmla_f32_z(predmr, C07, A0, B3);
	B3  = svdup_n_f32(Bptr[baseB + 3]); 	
	
	//----------------------------------------------
	A0 = svld1_f32(predmr, &Aptr[baseA]);
	baseA = baseA + MR;
	//----------------------------------------------

      }
    
      svst1_f32(predmr, &Ctref(0, 0),  C00);
      svst1_f32(predmr, &Ctref(0, 1),  C01);
      svst1_f32(predmr, &Ctref(0, 2),  C02);
      svst1_f32(predmr, &Ctref(0, 3),  C03);
      svst1_f32(predmr, &Ctref(0, 4),  C04);
      svst1_f32(predmr, &Ctref(0, 5),  C05);
      svst1_f32(predmr, &Ctref(0, 6),  C06);
      svst1_f32(predmr, &Ctref(0, 7),  C07);

    } else {
      //----------------------------------------------
      B0 = svdup_n_f32(Bptr[baseB + 0]); 
      B1 = svdup_n_f32(Bptr[baseB + 1]);
      B2 = svdup_n_f32(Bptr[baseB + 2]); 
      B3 = svdup_n_f32(Bptr[baseB + 3]);
      //----------------------------------------------
      
      //----------------------------------------------
      A0 = svld1_f32(pred16, &Aptr[baseA]);
      A1 = svld1_f32(pred16, &Aptr[baseA + 16]);
      baseA = baseA + MR;
      //----------------------------------------------
      
      //----------------------------------------------
      //Prefetching [C]
      //----------------------------------------------    
      svprfw(pred16, &Ccol(0, 0), SV_PLDL1STRM);
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
      svprfw(pred16, &Ccol(16, 7), SV_PLDL1STRM);
      //----------------------------------------------
      
      //------------------------
      //Clear columns
      //------------------------
      C00 = svdup_n_f32(0); 
      C01 = svdup_n_f32(0); 
      C02 = svdup_n_f32(0); 
      C03 = svdup_n_f32(0); 
      C04 = svdup_n_f32(0); 
      C05 = svdup_n_f32(0); 
      C06 = svdup_n_f32(0); 
      C07 = svdup_n_f32(0); 
      
      C10 = svdup_n_f32(0); 
      C11 = svdup_n_f32(0); 
      C12 = svdup_n_f32(0); 
      C13 = svdup_n_f32(0);
      C14 = svdup_n_f32(0); 
      C15 = svdup_n_f32(0);
      C16 = svdup_n_f32(0); 
      C17 = svdup_n_f32(0);
      //-------------------------
      
      for ( k=0; k < kc - 2; k += 2 ) {
	//----------------------------------------------
	A2 = svld1_f32(pred16, &Aptr[baseA]);
	A3 = svld1_f32(pred16, &Aptr[baseA + 16]);
	//----------------------------------------------
	
	C00 = svmla_f32_z(pred16, C00, A0, B0);
	C10 = svmla_f32_z(pred16, C10, A1, B0);
	B0  = svdup_n_f32(Bptr[baseB + 4]); 
	
	C01 = svmla_f32_z(pred16, C01, A0, B1);
	C11 = svmla_f32_z(pred16, C11, A1, B1);
	B1  = svdup_n_f32(Bptr[baseB + 5]);      
	
	C02 = svmla_f32_z(pred16, C02, A0, B2);
	C12 = svmla_f32_z(pred16, C12, A1, B2);
	B2  = svdup_n_f32(Bptr[baseB + 6]); 
      
	C03 = svmla_f32_z(pred16, C03, A0, B3);
	C13 = svmla_f32_z(pred16, C13, A1, B3);
	B3  = svdup_n_f32(Bptr[baseB + 7]); 

	baseB = baseB + NR;
	
	C04 = svmla_f32_z(pred16, C04, A0, B0);
	C14 = svmla_f32_z(pred16, C14, A1, B0);
	B0  = svdup_n_f32(Bptr[baseB + 0]); 
	
	C05 = svmla_f32_z(pred16, C05, A0, B1);
	C15 = svmla_f32_z(pred16, C15, A1, B1);
	B1  = svdup_n_f32(Bptr[baseB + 1]); 

	C06 = svmla_f32_z(pred16, C06, A0, B2);
	C16 = svmla_f32_z(pred16, C16, A1, B2);
	B2  = svdup_n_f32(Bptr[baseB + 2]); 
      
	C07 = svmla_f32_z(pred16, C07, A0, B3);
	C17 = svmla_f32_z(pred16, C17, A1, B3);
	B3  = svdup_n_f32(Bptr[baseB + 3]); 	
	
	//----------------------------------------------
	A0 = svld1_f32(pred16, &Aptr[baseA + 32]);
	A1 = svld1_f32(pred16, &Aptr[baseA + 48]);
	baseA = baseA + 64;
	//----------------------------------------------
	
	C00 = svmla_f32_z(pred16, C00, A2, B0);
	C10 = svmla_f32_z(pred16, C10, A3, B0);
	B0  = svdup_n_f32(Bptr[baseB + 4]); 
	
	C01 = svmla_f32_z(pred16, C01, A2, B1);
	C11 = svmla_f32_z(pred16, C11, A3, B1);
	B1  = svdup_n_f32(Bptr[baseB + 5]);      
	
	C02 = svmla_f32_z(pred16, C02, A2, B2);
	C12 = svmla_f32_z(pred16, C12, A3, B2);
	B2  = svdup_n_f32(Bptr[baseB + 6]); 
      
	C03 = svmla_f32_z(pred16, C03, A2, B3);
	C13 = svmla_f32_z(pred16, C13, A3, B3);
	B3  = svdup_n_f32(Bptr[baseB + 7]); 

	baseB = baseB + NR;
	
	C04 = svmla_f32_z(pred16, C04, A2, B0);
	C14 = svmla_f32_z(pred16, C14, A3, B0);
	B0  = svdup_n_f32(Bptr[baseB + 0]); 	
	
	C05 = svmla_f32_z(pred16, C05, A2, B1);
	C15 = svmla_f32_z(pred16, C15, A3, B1);
	B1  = svdup_n_f32(Bptr[baseB + 1]); 
	
	C06 = svmla_f32_z(pred16, C06, A2, B2);
	C16 = svmla_f32_z(pred16, C16, A3, B2);
	B2  = svdup_n_f32(Bptr[baseB + 2]); 
      
	C07 = svmla_f32_z(pred16, C07, A2, B3);
	C17 = svmla_f32_z(pred16, C17, A3, B3);
	B3  = svdup_n_f32(Bptr[baseB + 3]); 
      	
      }
      
      C00 = svmla_f32_z(pred16, C00, A0, B0);
      C10 = svmla_f32_z(pred16, C10, A1, B0);
      B0  = svdup_n_f32(Bptr[baseB + 4]); 
      
      C01 = svmla_f32_z(pred16, C01, A0, B1);
      C11 = svmla_f32_z(pred16, C11, A1, B1);
      B1  = svdup_n_f32(Bptr[baseB + 5]);      
      
      C02 = svmla_f32_z(pred16, C02, A0, B2);
      C12 = svmla_f32_z(pred16, C12, A1, B2);
      B2  = svdup_n_f32(Bptr[baseB + 6]); 
      
      C03 = svmla_f32_z(pred16, C03, A0, B3);
      C13 = svmla_f32_z(pred16, C13, A1, B3);
      B3  = svdup_n_f32(Bptr[baseB + 7]); 
      
      C04 = svmla_f32_z(pred16, C04, A0, B0);
      C14 = svmla_f32_z(pred16, C14, A1, B0);
	
      C05 = svmla_f32_z(pred16, C05, A0, B1);
      C15 = svmla_f32_z(pred16, C15, A1, B1);
      
      C06 = svmla_f32_z(pred16, C06, A0, B2);
      C16 = svmla_f32_z(pred16, C16, A1, B2);
      
      C07 = svmla_f32_z(pred16, C07, A0, B3);
      C17 = svmla_f32_z(pred16, C17, A1, B3);
      
      
      if (kc % 2 == 0) {	
	baseB = baseB + NR;	
	
	//----------------------------------------------
	A2 = svld1_f32(pred16, &Aptr[baseA]);
	A3 = svld1_f32(pred16, &Aptr[baseA + 16]);
	//----------------------------------------------
	
	//----------------------------------------------
	B0 = svdup_n_f32(Bptr[baseB + 0]); 
	B1 = svdup_n_f32(Bptr[baseB + 1]);
	B2 = svdup_n_f32(Bptr[baseB + 2]); 
	B3 = svdup_n_f32(Bptr[baseB + 3]);
	//----------------------------------------------
	
	C00 = svmla_f32_z(pred16, C00, A2, B0);
	C10 = svmla_f32_z(pred16, C10, A3, B0);
	B0  = svdup_n_f32(Bptr[baseB + 4]); 
	
	C01 = svmla_f32_z(pred16, C01, A2, B1);
	C11 = svmla_f32_z(pred16, C11, A3, B1);
	B1  = svdup_n_f32(Bptr[baseB + 5]);      
	
	C02 = svmla_f32_z(pred16, C02, A2, B2);
	C12 = svmla_f32_z(pred16, C12, A3, B2);
	B2  = svdup_n_f32(Bptr[baseB + 6]); 
	
	C03 = svmla_f32_z(pred16, C03, A2, B3);
	C13 = svmla_f32_z(pred16, C13, A3, B3);
	B3  = svdup_n_f32(Bptr[baseB + 7]); 
	
	C04 = svmla_f32_z(pred16, C04, A2, B0);
	C14 = svmla_f32_z(pred16, C14, A3, B0);
	
	C05 = svmla_f32_z(pred16, C05, A2, B1);
	C15 = svmla_f32_z(pred16, C15, A3, B1);
	
	C06 = svmla_f32_z(pred16, C06, A2, B2);
	C16 = svmla_f32_z(pred16, C16, A3, B2);
	
	C07 = svmla_f32_z(pred16, C07, A2, B3);
	C17 = svmla_f32_z(pred16, C17, A3, B3);
		
      }
      
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

    }

  } else {
    if (mr <= 16) {
      
      svbool_t predmr = svwhilelt_b32_u32(0, mr);

      //----------------------------------------------
      B0 = svdup_n_f32(Bptr[baseB + 0]); 
      B1 = svdup_n_f32(Bptr[baseB + 1]);
      B2 = svdup_n_f32(Bptr[baseB + 2]); 
      B3 = svdup_n_f32(Bptr[baseB + 3]);
      B4 = svdup_n_f32(Bptr[baseB + 4]);
      //----------------------------------------------
      
      //----------------------------------------------
      A0 = svld1_f32(predmr, &Aptr[baseA]);
      baseA = baseA + MR;
      //----------------------------------------------
      
      //----------------------------------------------
      //Prefetching [C]
      //----------------------------------------------    
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
      //----------------------------------------------
      
      //------------------------
      //Clear columns
      //------------------------
      C00 = svdup_n_f32(0); 
      C01 = svdup_n_f32(0); 
      C02 = svdup_n_f32(0); 
      C03 = svdup_n_f32(0); 
      C04 = svdup_n_f32(0); 
      C05 = svdup_n_f32(0); 
      C06 = svdup_n_f32(0); 
      C07 = svdup_n_f32(0); 
      C08 = svdup_n_f32(0); 
      C09 = svdup_n_f32(0);     
      //-------------------------
      
      for ( k=0; k < kc; k ++) {
	
	C00 = svmla_f32_z(predmr, C00, A0, B0);
	B0  = svdup_n_f32(Bptr[baseB + 5]); 
      
	C01 = svmla_f32_z(predmr, C01, A0, B1);
	B1  = svdup_n_f32(Bptr[baseB + 6]);      
      
	C02 = svmla_f32_z(predmr, C02, A0, B2);
	B2  = svdup_n_f32(Bptr[baseB + 7]); 
      
	C03 = svmla_f32_z(predmr, C03, A0, B3);
	B3  = svdup_n_f32(Bptr[baseB + 8]); 
      
	C04 = svmla_f32_z(predmr, C04, A0, B4);
	B4  = svdup_n_f32(Bptr[baseB + 9]); 

	baseB = baseB + NR;
	
	C05 = svmla_f32_z(predmr, C05, A0, B0);
	B0  = svdup_n_f32(Bptr[baseB + 0]); 
	
	C06 = svmla_f32_z(predmr, C06, A0, B1);
	B1  = svdup_n_f32(Bptr[baseB + 1]); 
      
	C07 = svmla_f32_z(predmr, C07, A0, B2);
	B2  = svdup_n_f32(Bptr[baseB + 2]); 	
	
	C08 = svmla_f32_z(predmr, C08, A0, B3);
	B3  = svdup_n_f32(Bptr[baseB + 3]); 
	
	C09 = svmla_f32_z(predmr, C09, A0, B4);
	B4  = svdup_n_f32(Bptr[baseB + 4]);       	

	//----------------------------------------------
	A0 = svld1_f32(predmr, &Aptr[baseA]);
	baseA = baseA + MR;
	//----------------------------------------------

      }
    
      svst1_f32(predmr, &Ctref(0, 0),  C00);
      svst1_f32(predmr, &Ctref(0, 1),  C01);
      svst1_f32(predmr, &Ctref(0, 2),  C02);
      svst1_f32(predmr, &Ctref(0, 3),  C03);
      svst1_f32(predmr, &Ctref(0, 4),  C04);
      svst1_f32(predmr, &Ctref(0, 5),  C05);
      svst1_f32(predmr, &Ctref(0, 6),  C06);
      svst1_f32(predmr, &Ctref(0, 7),  C07);
      svst1_f32(predmr, &Ctref(0, 8),  C08);
      svst1_f32(predmr, &Ctref(0, 9),  C09);

    } else {
      //----------------------------------------------
      B0 = svdup_n_f32(Bptr[baseB + 0]); 
      B1 = svdup_n_f32(Bptr[baseB + 1]);
      B2 = svdup_n_f32(Bptr[baseB + 2]); 
      B3 = svdup_n_f32(Bptr[baseB + 3]);
      B4 = svdup_n_f32(Bptr[baseB + 4]);
      //----------------------------------------------
      
      //----------------------------------------------
      A0 = svld1_f32(pred16, &Aptr[baseA]);
      A1 = svld1_f32(pred16, &Aptr[baseA + 16]);
      baseA = baseA + MR;
      //----------------------------------------------
      
      //----------------------------------------------
      //Prefetching [C]
      //----------------------------------------------    
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
      //----------------------------------------------
      
      //------------------------
      //Clear columns
      //------------------------
      C00 = svdup_n_f32(0); 
      C01 = svdup_n_f32(0); 
      C02 = svdup_n_f32(0); 
      C03 = svdup_n_f32(0); 
      C04 = svdup_n_f32(0); 
      C05 = svdup_n_f32(0); 
      C06 = svdup_n_f32(0); 
      C07 = svdup_n_f32(0); 
      C08 = svdup_n_f32(0); 
      C09 = svdup_n_f32(0); 
      
      C10 = svdup_n_f32(0); 
      C11 = svdup_n_f32(0); 
      C12 = svdup_n_f32(0); 
      C13 = svdup_n_f32(0);
      C14 = svdup_n_f32(0); 
      C15 = svdup_n_f32(0);
      C16 = svdup_n_f32(0); 
      C17 = svdup_n_f32(0);
      C18 = svdup_n_f32(0); 
      C19 = svdup_n_f32(0); 
      //-------------------------
      
      for ( k=0; k < kc - 2; k += 2 ) {
	//----------------------------------------------
	A2 = svld1_f32(pred16, &Aptr[baseA]);
	A3 = svld1_f32(pred16, &Aptr[baseA + 16]);
	//----------------------------------------------
	
	C00 = svmla_f32_z(pred16, C00, A0, B0);
	C10 = svmla_f32_z(pred16, C10, A1, B0);
	B0  = svdup_n_f32(Bptr[baseB + 5]); 
	
	C01 = svmla_f32_z(pred16, C01, A0, B1);
	C11 = svmla_f32_z(pred16, C11, A1, B1);
	B1  = svdup_n_f32(Bptr[baseB + 6]);      
	
	C02 = svmla_f32_z(pred16, C02, A0, B2);
	C12 = svmla_f32_z(pred16, C12, A1, B2);
	B2  = svdup_n_f32(Bptr[baseB + 7]); 
      
	C03 = svmla_f32_z(pred16, C03, A0, B3);
	C13 = svmla_f32_z(pred16, C13, A1, B3);
	B3  = svdup_n_f32(Bptr[baseB + 8]); 
      
	C04 = svmla_f32_z(pred16, C04, A0, B4);
	C14 = svmla_f32_z(pred16, C14, A1, B4);
	B4  = svdup_n_f32(Bptr[baseB + 9]); 

	baseB = baseB + NR;
	
	C05 = svmla_f32_z(pred16, C05, A0, B0);
	C15 = svmla_f32_z(pred16, C15, A1, B0);
	B0  = svdup_n_f32(Bptr[baseB + 0]); 

	C06 = svmla_f32_z(pred16, C06, A0, B1);
	C16 = svmla_f32_z(pred16, C16, A1, B1);
	B1  = svdup_n_f32(Bptr[baseB + 1]); 
      
	C07 = svmla_f32_z(pred16, C07, A0, B2);
	C17 = svmla_f32_z(pred16, C17, A1, B2);
	B2  = svdup_n_f32(Bptr[baseB + 2]); 	
      
	C08 = svmla_f32_z(pred16, C08, A0, B3);
	C18 = svmla_f32_z(pred16, C18, A1, B3);
	B3  = svdup_n_f32(Bptr[baseB + 3]); 
      
	C09 = svmla_f32_z(pred16, C09, A0, B4);
	C19 = svmla_f32_z(pred16, C19, A1, B4);
	B4  = svdup_n_f32(Bptr[baseB + 4]);       
	
	//----------------------------------------------
	A0 = svld1_f32(pred16, &Aptr[baseA + 32]);
	A1 = svld1_f32(pred16, &Aptr[baseA + 48]);
	baseA = baseA + 64;
	//----------------------------------------------
	
	C00 = svmla_f32_z(pred16, C00, A2, B0);
	C10 = svmla_f32_z(pred16, C10, A3, B0);
	B0  = svdup_n_f32(Bptr[baseB + 5]); 
      
	C01 = svmla_f32_z(pred16, C01, A2, B1);
	C11 = svmla_f32_z(pred16, C11, A3, B1);
	B1  = svdup_n_f32(Bptr[baseB + 6]);      
      
	C02 = svmla_f32_z(pred16, C02, A2, B2);
	C12 = svmla_f32_z(pred16, C12, A3, B2);
	B2  = svdup_n_f32(Bptr[baseB + 7]); 
      
	C03 = svmla_f32_z(pred16, C03, A2, B3);
	C13 = svmla_f32_z(pred16, C13, A3, B3);
	B3  = svdup_n_f32(Bptr[baseB + 8]); 
	
	C04 = svmla_f32_z(pred16, C04, A2, B4);
	C14 = svmla_f32_z(pred16, C14, A3, B4);
	B4  = svdup_n_f32(Bptr[baseB + 9]); 
	
	baseB = baseB + NR;
	
	C05 = svmla_f32_z(pred16, C05, A2, B0);
	C15 = svmla_f32_z(pred16, C15, A3, B0);
	B0  = svdup_n_f32(Bptr[baseB + 0]); 
	
	C06 = svmla_f32_z(pred16, C06, A2, B1);
	C16 = svmla_f32_z(pred16, C16, A3, B1);
	B1  = svdup_n_f32(Bptr[baseB + 1]); 
      
	C07 = svmla_f32_z(pred16, C07, A2, B2);
	C17 = svmla_f32_z(pred16, C17, A3, B2);
	B2  = svdup_n_f32(Bptr[baseB + 2]); 
      
	C08 = svmla_f32_z(pred16, C08, A2, B3);
	C18 = svmla_f32_z(pred16, C18, A3, B3);
	B3  = svdup_n_f32(Bptr[baseB + 3]); 
	
	C09 = svmla_f32_z(pred16, C09, A2, B4);
	C19 = svmla_f32_z(pred16, C19, A3, B4);
	B4  = svdup_n_f32(Bptr[baseB + 4]);       
	
      }
      
      C00 = svmla_f32_z(pred16, C00, A0, B0);
      C10 = svmla_f32_z(pred16, C10, A1, B0);
      B0  = svdup_n_f32(Bptr[baseB + 5]); 
      
      C01 = svmla_f32_z(pred16, C01, A0, B1);
      C11 = svmla_f32_z(pred16, C11, A1, B1);
      B1  = svdup_n_f32(Bptr[baseB + 6]);      
      
      C02 = svmla_f32_z(pred16, C02, A0, B2);
      C12 = svmla_f32_z(pred16, C12, A1, B2);
      B2  = svdup_n_f32(Bptr[baseB + 7]); 
      
      C03 = svmla_f32_z(pred16, C03, A0, B3);
      C13 = svmla_f32_z(pred16, C13, A1, B3);
      B3  = svdup_n_f32(Bptr[baseB + 8]); 
      
      C04 = svmla_f32_z(pred16, C04, A0, B4);
      C14 = svmla_f32_z(pred16, C14, A1, B4);
      B4  = svdup_n_f32(Bptr[baseB + 9]); 
	
      C05 = svmla_f32_z(pred16, C05, A0, B0);
      C15 = svmla_f32_z(pred16, C15, A1, B0);
      
      C06 = svmla_f32_z(pred16, C06, A0, B1);
      C16 = svmla_f32_z(pred16, C16, A1, B1);
      
      C07 = svmla_f32_z(pred16, C07, A0, B2);
      C17 = svmla_f32_z(pred16, C17, A1, B2);
      
      C08 = svmla_f32_z(pred16, C08, A0, B3);
      C18 = svmla_f32_z(pred16, C18, A1, B3);
      
      C09 = svmla_f32_z(pred16, C09, A0, B4);
      C19 = svmla_f32_z(pred16, C19, A1, B4);
      
      if (kc % 2 == 0) {	
	baseB = baseB + NR;	

	//----------------------------------------------
	A2 = svld1_f32(pred16, &Aptr[baseA]);
	A3 = svld1_f32(pred16, &Aptr[baseA + 16]);
	//----------------------------------------------
	
	//----------------------------------------------
	B0 = svdup_n_f32(Bptr[baseB + 0]); 
	B1 = svdup_n_f32(Bptr[baseB + 1]);
	B2 = svdup_n_f32(Bptr[baseB + 2]); 
	B3 = svdup_n_f32(Bptr[baseB + 3]);
	B4 = svdup_n_f32(Bptr[baseB + 4]);
	//----------------------------------------------
	
	C00 = svmla_f32_z(pred16, C00, A2, B0);
	C10 = svmla_f32_z(pred16, C10, A3, B0);
	B0  = svdup_n_f32(Bptr[baseB + 5]); 
	
	C01 = svmla_f32_z(pred16, C01, A2, B1);
	C11 = svmla_f32_z(pred16, C11, A3, B1);
	B1  = svdup_n_f32(Bptr[baseB + 6]);      
	
	C02 = svmla_f32_z(pred16, C02, A2, B2);
	C12 = svmla_f32_z(pred16, C12, A3, B2);
	B2  = svdup_n_f32(Bptr[baseB + 7]); 
	
	C03 = svmla_f32_z(pred16, C03, A2, B3);
	C13 = svmla_f32_z(pred16, C13, A3, B3);
	B3  = svdup_n_f32(Bptr[baseB + 8]); 
	
	C04 = svmla_f32_z(pred16, C04, A2, B4);
	C14 = svmla_f32_z(pred16, C14, A3, B4);
	B4  = svdup_n_f32(Bptr[baseB + 9]); 
	
	C05 = svmla_f32_z(pred16, C05, A2, B0);
	C15 = svmla_f32_z(pred16, C15, A3, B0);
	
	C06 = svmla_f32_z(pred16, C06, A2, B1);
	C16 = svmla_f32_z(pred16, C16, A3, B1);
	
	C07 = svmla_f32_z(pred16, C07, A2, B2);
	C17 = svmla_f32_z(pred16, C17, A3, B2);
	
	C08 = svmla_f32_z(pred16, C08, A2, B3);
	C18 = svmla_f32_z(pred16, C18, A3, B3);
	
	C09 = svmla_f32_z(pred16, C09, A2, B4);
	C19 = svmla_f32_z(pred16, C19, A3, B4);
	
      }
      
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
    }

    /*
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
    ACOL_LOAD_32X10(A0, A1, pred16, Aptr, baseA);
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
    */
  }
  
  if ( beta != zero )
    for ( j=0; j < nr; j++ )
      for ( i=0; i < mr; i++ )
	Ccol(i,j) = Ccol(i,j) + Ctcol(i,j);
  else
    for ( j=0; j<nr; j++ )
      for ( i=0; i<mr; i++ )
	Ccol(i,j) = Ctcol(i,j);  
  
}


//======================================================================================
//**************************************************************************************
//*                       M I C R O - K E R N E L    32 X 10                            *
//**************************************************************************************
//======================================================================================






void gemm_microkernel_sve_10x32_fp32( int mr, int nr, int kc, float alpha, float *Ar, float *Br, float beta, float *C, int ldC ) {
/*
  BLIS GEMM microkernel, computes the product Cr := Cr + Ar * Br
  Specific: only for MRxNR = 10x32
*/
  //SET_MR_NR(10, 32);
  
  int         i, j, k, baseB = 0, baseA = 0, ldCt = NR, Amr, Bnr;

  //Registers for C = (20)
  svfloat32_t C00, C10, C20, C30, C40, C50, C60, C70, C80, C90,
              C01, C11, C21, C31, C41, C51, C61, C71, C81, C91;

  //Other auxiliar Registers (12)
  svfloat32_t  B0,  B1, B2, B3, A0, A1, A2, A3, A4, A5, A6, A7;
  
  float   zero = 0.0, one = 1.0, *Aptr, *Bptr, Ctmp[MR * NR];
  svbool_t pred16 = svwhilelt_b32_u32(0, 16);
  
  if ( kc==0 )
    return;
    
  Aptr = &Ar[0];
  Amr  = MR;
  Bptr = &Br[0];
  Bnr  = NR;
  
  if ((mr <= 2) && (nr <= 32)) {
    //Prefetching B
    B0 = svld1_f32(pred16, &Bptr[baseB]);
    B1 = svld1_f32(pred16, &Bptr[baseB + 16]);
    
    //Prefetching A
    A0 = svdup_n_f32(Aptr[baseA + 0]);
    A1 = svdup_n_f32(Aptr[baseA + 1]);
    
    baseA = baseA + Amr;
    baseB = baseB + Bnr;      
    
    C00 = svld1_f32(pred16, &Crow(0, 0));
    C10 = svld1_f32(pred16, &Crow(1, 0));

    C01 = svld1_f32(pred16, &Crow(0, 16));
    C11 = svld1_f32(pred16, &Crow(1, 16));
    
    for ( k=0; k < kc - 2; k += 2 ) {
      //Iter (+1)
      B2 = svld1_f32(pred16, &Bptr[baseB]);
      B3 = svld1_f32(pred16, &Bptr[baseB + 16]);
      
      C00 = svmla_f32_z(pred16, C00, B0, A0);
      C01 = svmla_f32_z(pred16, C01, B1, A0);
      A0=svdup_n_f32(Aptr[baseA + 0]);
      
      C10 = svmla_f32_z(pred16, C10, B0, A1);
      C11 = svmla_f32_z(pred16, C11, B1, A1);
      A1=svdup_n_f32(Aptr[baseA + 1]);
      
      //Iter (+2)
      B0 = svld1_f32(pred16, &Bptr[baseB + 32]);
      B1 = svld1_f32(pred16, &Bptr[baseB + 48]);
      
      C00 = svmla_f32_z(pred16, C00, B2, A0);
      C01 = svmla_f32_z(pred16, C01, B3, A0);
      A0=svdup_n_f32(Aptr[baseA + 10]);
      
      C10 = svmla_f32_z(pred16, C10, B2, A1);
      C11 = svmla_f32_z(pred16, C11, B3, A1);
      A1=svdup_n_f32(Aptr[baseA + 11]);
      
      baseA = baseA + 20;
      baseB = baseB + 64;      
    }
    
    C00 = svmla_f32_z(pred16, C00, B0, A0);
    C01 = svmla_f32_z(pred16, C01, B1, A0);

    C10 = svmla_f32_z(pred16, C10, B0, A1);
    C11 = svmla_f32_z(pred16, C11, B1, A1);

    if (kc % 2 == 0) {
      B0 = svld1_f32(pred16, &Bptr[baseB]);
      B1 = svld1_f32(pred16, &Bptr[baseB + 16]);
      
      A0 = svdup_n_f32(Aptr[baseA + 0]);      
      C00 = svmla_f32_z(pred16, C00, B0, A0);
      A1  = svdup_n_f32(Aptr[baseA + 1]);
      C01 = svmla_f32_z(pred16, C01, B1, A0);
      C10 = svmla_f32_z(pred16, C10, B0, A1);
      C11 = svmla_f32_z(pred16, C11, B1, A1);
    }
    
    svst1_f32(pred16, &Ctrow(0, 0),  C00);
    svst1_f32(pred16, &Ctrow(1, 0),  C10);    
    
    svst1_f32(pred16, &Ctrow(0, 16), C01);
    svst1_f32(pred16, &Ctrow(1, 16), C11);
    
    for ( j=0; j<nr; j++ )
      for ( i=0; i<mr; i++ )
	Crow(i,j) = Ctrow(i, j);
    
  } else if ((mr <= 4) && (nr <= 32)) {
    
    //Prefetching B
    B0 = svld1_f32(pred16, &Bptr[baseB]);
    B1 = svld1_f32(pred16, &Bptr[baseB + 16]);
    
    //Prefetching A
    A0 = svdup_n_f32(Aptr[baseA + 0]);
    A1 = svdup_n_f32(Aptr[baseA + 1]);
    A2 = svdup_n_f32(Aptr[baseA + 2]);
    A3 = svdup_n_f32(Aptr[baseA + 3]);
    
    baseA = baseA + Amr;
    baseB = baseB + Bnr;      
    
    C00 = svld1_f32(pred16, &Crow(0, 0));
    C10 = svld1_f32(pred16, &Crow(1, 0));
    C20 = svld1_f32(pred16, &Crow(2, 0));
    C30 = svld1_f32(pred16, &Crow(3, 0));
    
    C01 = svld1_f32(pred16, &Crow(0, 16));
    C11 = svld1_f32(pred16, &Crow(1, 16));
    C21 = svld1_f32(pred16, &Crow(2, 16));
    C31 = svld1_f32(pred16, &Crow(3, 16));
    
    for ( k=0; k < kc - 2; k += 2 ) {
      //Iter (+1)
      B2 = svld1_f32(pred16, &Bptr[baseB]);
      B3 = svld1_f32(pred16, &Bptr[baseB + 16]);
      
      C00 = svmla_f32_z(pred16, C00, B0, A0);
      C01 = svmla_f32_z(pred16, C01, B1, A0);
      A0=svdup_n_f32(Aptr[baseA + 0]);
      
      C10 = svmla_f32_z(pred16, C10, B0, A1);
      C11 = svmla_f32_z(pred16, C11, B1, A1);
      A1=svdup_n_f32(Aptr[baseA + 1]);
      
      C20 = svmla_f32_z(pred16, C20, B0, A2);
      C21 = svmla_f32_z(pred16, C21, B1, A2);
      A2=svdup_n_f32(Aptr[baseA + 2]);
      
      C30 = svmla_f32_z(pred16, C30, B0, A3);
      C31 = svmla_f32_z(pred16, C31, B1, A3);
      A3=svdup_n_f32(Aptr[baseA + 3]);
      
      //Iter (+2)
      B0 = svld1_f32(pred16, &Bptr[baseB + 32]);
      B1 = svld1_f32(pred16, &Bptr[baseB + 48]);
      
      C00 = svmla_f32_z(pred16, C00, B2, A0);
      C01 = svmla_f32_z(pred16, C01, B3, A0);
      A0=svdup_n_f32(Aptr[baseA + 10]);
      
      C10 = svmla_f32_z(pred16, C10, B2, A1);
      C11 = svmla_f32_z(pred16, C11, B3, A1);
      A1=svdup_n_f32(Aptr[baseA + 11]);
      
      C20 = svmla_f32_z(pred16, C20, B2, A2);
      C21 = svmla_f32_z(pred16, C21, B3, A2);
      A2=svdup_n_f32(Aptr[baseA + 12]);
      
      C30 = svmla_f32_z(pred16, C30, B2, A3);
      C31 = svmla_f32_z(pred16, C31, B3, A3);
      A3=svdup_n_f32(Aptr[baseA + 13]);
      
      baseA = baseA + 20;
      baseB = baseB + 64;      
    }
    
    
    C00 = svmla_f32_z(pred16, C00, B0, A0);
    C01 = svmla_f32_z(pred16, C01, B1, A0);
    C10 = svmla_f32_z(pred16, C10, B0, A1);
    C11 = svmla_f32_z(pred16, C11, B1, A1);
    C20 = svmla_f32_z(pred16, C20, B0, A2);
    C21 = svmla_f32_z(pred16, C21, B1, A2);
    C30 = svmla_f32_z(pred16, C30, B0, A3);
    C31 = svmla_f32_z(pred16, C31, B1, A3);
    
    if (kc % 2 == 0) {
      B0 = svld1_f32(pred16, &Bptr[baseB]);
      B1 = svld1_f32(pred16, &Bptr[baseB + 16]);
      
      A0 = svdup_n_f32(Aptr[baseA + 0]);      
      C00 = svmla_f32_z(pred16, C00, B0, A0);
      A1  = svdup_n_f32(Aptr[baseA + 1]);
      C01 = svmla_f32_z(pred16, C01, B1, A0);
      A2  = svdup_n_f32(Aptr[baseA + 2]);
      C10 = svmla_f32_z(pred16, C10, B0, A1);
      A3  = svdup_n_f32(Aptr[baseA + 3]);
      C11 = svmla_f32_z(pred16, C11, B1, A1);
      C20 = svmla_f32_z(pred16, C20, B0, A2);
      C21 = svmla_f32_z(pred16, C21, B1, A2);
      C30 = svmla_f32_z(pred16, C30, B0, A3);
      C31 = svmla_f32_z(pred16, C31, B1, A3);
    }

    svst1_f32(pred16, &Ctrow(0, 0),  C00);
    svst1_f32(pred16, &Ctrow(1, 0),  C10);    
    svst1_f32(pred16, &Ctrow(2, 0),  C20);
    svst1_f32(pred16, &Ctrow(3, 0),  C30);
    
    svst1_f32(pred16, &Ctrow(0, 16), C01);
    svst1_f32(pred16, &Ctrow(1, 16), C11);
    svst1_f32(pred16, &Ctrow(2, 16), C21);
    svst1_f32(pred16, &Ctrow(3, 16), C31);

    for ( j=0; j<nr; j++ )
      for ( i=0; i<mr; i++ )
	Crow(i,j) = Ctrow(i, j);

  } else if ((mr <= 6) && (nr == 32)) {      
    
    //Prefetching B
    B0 = svld1_f32(pred16, &Bptr[baseB]);
    B1 = svld1_f32(pred16, &Bptr[baseB + 16]);
    
    //Prefetching A
    A0 = svdup_n_f32(Aptr[baseA + 0]);
    A1 = svdup_n_f32(Aptr[baseA + 1]);
    A2 = svdup_n_f32(Aptr[baseA + 2]);
    A3 = svdup_n_f32(Aptr[baseA + 3]);
    A4 = svdup_n_f32(Aptr[baseA + 4]);
    A5 = svdup_n_f32(Aptr[baseA + 5]);
    
    baseA = baseA + Amr;
    baseB = baseB + Bnr;      
    
    C00 = svld1_f32(pred16, &Crow(0, 0));
    C10 = svld1_f32(pred16, &Crow(1, 0));
    C20 = svld1_f32(pred16, &Crow(2, 0));
    C30 = svld1_f32(pred16, &Crow(3, 0));
    C40 = svld1_f32(pred16, &Crow(4, 0));
    C50 = svld1_f32(pred16, &Crow(5, 0));
    
    C01 = svld1_f32(pred16, &Crow(0, 16));
    C11 = svld1_f32(pred16, &Crow(1, 16));
    C21 = svld1_f32(pred16, &Crow(2, 16));
    C31 = svld1_f32(pred16, &Crow(3, 16));
    C41 = svld1_f32(pred16, &Crow(4, 16));
    C51 = svld1_f32(pred16, &Crow(5, 16));
    
    for ( k = 0; k < kc - 2; k += 2 ) {

      //Iter (+1)
      B2 = svld1_f32(pred16, &Bptr[baseB ]);
      B3 = svld1_f32(pred16, &Bptr[baseB + 16]);
      
      C00 = svmla_f32_z(pred16, C00, B0, A0);
      C01 = svmla_f32_z(pred16, C01, B1, A0);
      A0  = svdup_n_f32(Aptr[baseA + 0]);
      
      C10 = svmla_f32_z(pred16, C10, B0, A1);
      C11 = svmla_f32_z(pred16, C11, B1, A1);
      A1  = svdup_n_f32(Aptr[baseA + 1]);
      
      C20 = svmla_f32_z(pred16, C20, B0, A2);
      C21 = svmla_f32_z(pred16, C21, B1, A2);
      A2  = svdup_n_f32(Aptr[baseA + 2]);
      
      C30 = svmla_f32_z(pred16, C30, B0, A3);
      C31 = svmla_f32_z(pred16, C31, B1, A3);
      A3  = svdup_n_f32(Aptr[baseA + 3]);
      
      C40 = svmla_f32_z(pred16, C40, B0, A4);
      C41 = svmla_f32_z(pred16, C41, B1, A4);
      A4  = svdup_n_f32(Aptr[baseA + 4]);
      
      C50 = svmla_f32_z(pred16, C50, B0, A5);
      C51 = svmla_f32_z(pred16, C51, B1, A5);
      A5  = svdup_n_f32(Aptr[baseA + 5]);
      
      
      //Iter (+2)
      B0 = svld1_f32(pred16, &Bptr[baseB + 32]);
      B1 = svld1_f32(pred16, &Bptr[baseB + 48]);
      
      C00 = svmla_f32_z(pred16, C00, B2, A0);
      C01 = svmla_f32_z(pred16, C01, B3, A0);
      A0  = svdup_n_f32(Aptr[baseA + 10]);
      
      C10 = svmla_f32_z(pred16, C10, B2, A1);
      C11 = svmla_f32_z(pred16, C11, B3, A1);
      A1  = svdup_n_f32(Aptr[baseA + 11]);
      
      C20 = svmla_f32_z(pred16, C20, B2, A2);
      C21 = svmla_f32_z(pred16, C21, B3, A2);
      A2  = svdup_n_f32(Aptr[baseA + 12]);
      
      C30 = svmla_f32_z(pred16, C30, B2, A3);
      C31 = svmla_f32_z(pred16, C31, B3, A3);
      A3  = svdup_n_f32(Aptr[baseA + 13]);
      
      C40 = svmla_f32_z(pred16, C40, B2, A4);
      C41 = svmla_f32_z(pred16, C41, B3, A4);
      A4  = svdup_n_f32(Aptr[baseA + 14]);
      
      C50 = svmla_f32_z(pred16, C50, B2, A5);
      C51 = svmla_f32_z(pred16, C51, B3, A5);
      A5  = svdup_n_f32(Aptr[baseA + 15]);      
            
      baseA = baseA + 20;
      baseB = baseB + 64;
    }

    C00 = svmla_f32_z(pred16, C00, B0, A0);
    C01 = svmla_f32_z(pred16, C01, B1, A0);
    C10 = svmla_f32_z(pred16, C10, B0, A1);
    C11 = svmla_f32_z(pred16, C11, B1, A1);
    C20 = svmla_f32_z(pred16, C20, B0, A2);
    C21 = svmla_f32_z(pred16, C21, B1, A2);
    C30 = svmla_f32_z(pred16, C30, B0, A3);
    C31 = svmla_f32_z(pred16, C31, B1, A3);
    C40 = svmla_f32_z(pred16, C40, B0, A4);
    C41 = svmla_f32_z(pred16, C41, B1, A4);
    C50 = svmla_f32_z(pred16, C50, B0, A5);
    C51 = svmla_f32_z(pred16, C51, B1, A5);
    
    if (kc % 2 == 0) {
      B0 = svld1_f32(pred16, &Bptr[baseB]);
      B1 = svld1_f32(pred16, &Bptr[baseB + 16]);
      
      A0 = svdup_n_f32(Aptr[baseA + 0]);      
      C00 = svmla_f32_z(pred16, C00, B0, A0);
      A1  = svdup_n_f32(Aptr[baseA + 1]);
      C01 = svmla_f32_z(pred16, C01, B1, A0);
      A2  = svdup_n_f32(Aptr[baseA + 2]);
      C10 = svmla_f32_z(pred16, C10, B0, A1);
      A3  = svdup_n_f32(Aptr[baseA + 3]);
      C11 = svmla_f32_z(pred16, C11, B1, A1);
      A4  = svdup_n_f32(Aptr[baseA + 4]);
      C20 = svmla_f32_z(pred16, C20, B0, A2);
      A5  = svdup_n_f32(Aptr[baseA + 5]);
      C21 = svmla_f32_z(pred16, C21, B1, A2);
      C30 = svmla_f32_z(pred16, C30, B0, A3);
      C31 = svmla_f32_z(pred16, C31, B1, A3);
      C40 = svmla_f32_z(pred16, C40, B0, A4);
      C41 = svmla_f32_z(pred16, C41, B1, A4);
      C50 = svmla_f32_z(pred16, C50, B0, A5);
      C51 = svmla_f32_z(pred16, C51, B1, A5);
    }
    
    svst1_f32(pred16, &Ctrow(0, 0),  C00);
    svst1_f32(pred16, &Ctrow(1, 0),  C10);    
    svst1_f32(pred16, &Ctrow(2, 0),  C20);
    svst1_f32(pred16, &Ctrow(3, 0),  C30); 
    svst1_f32(pred16, &Ctrow(4, 0),  C40);
    svst1_f32(pred16, &Ctrow(5, 0),  C50); 

    svst1_f32(pred16, &Ctrow(0, 16), C01);
    svst1_f32(pred16, &Ctrow(1, 16), C11);
    svst1_f32(pred16, &Ctrow(2, 16), C21);
    svst1_f32(pred16, &Ctrow(3, 16), C31);
    svst1_f32(pred16, &Ctrow(4, 16), C41);
    svst1_f32(pred16, &Ctrow(5, 16), C51);
    
    for ( j=0; j<nr; j++ )
      for ( i=0; i<mr; i++ )
	Crow(i, j) = Ctrow(i, j);
    
  } else if ((mr <= 8) && (nr <= 32)) {
        
    C00 = svld1_f32(pred16, &Crow(0, 0));
    C10 = svld1_f32(pred16, &Crow(1, 0));
    C20 = svld1_f32(pred16, &Crow(2, 0));
    C30 = svld1_f32(pred16, &Crow(3, 0));
    C40 = svld1_f32(pred16, &Crow(4, 0));
    C50 = svld1_f32(pred16, &Crow(5, 0));
    C60 = svld1_f32(pred16, &Crow(6, 0));
    C70 = svld1_f32(pred16, &Crow(7, 0));
    
    C01 = svld1_f32(pred16, &Crow(0, 16));
    C11 = svld1_f32(pred16, &Crow(1, 16));
    C21 = svld1_f32(pred16, &Crow(2, 16));
    C31 = svld1_f32(pred16, &Crow(3, 16));
    C41 = svld1_f32(pred16, &Crow(4, 16));
    C51 = svld1_f32(pred16, &Crow(5, 16));
    C61 = svld1_f32(pred16, &Crow(6, 16));
    C71 = svld1_f32(pred16, &Crow(7, 16));

    for ( k = 0; k < kc; k++ ) {

      //Prefetching B
      B0 = svld1_f32(pred16, &Bptr[baseB]);
      B1 = svld1_f32(pred16, &Bptr[baseB + 16]);

      A0  = svdup_n_f32(Aptr[baseA + 0]);      
      C00 = svmla_f32_z(pred16, C00, B0, A0);
      C01 = svmla_f32_z(pred16, C01, B1, A0);      

      A1  = svdup_n_f32(Aptr[baseA + 1]);
      C10 = svmla_f32_z(pred16, C10, B0, A1);
      C11 = svmla_f32_z(pred16, C11, B1, A1);

      A0  = svdup_n_f32(Aptr[baseA + 2]);      
      C20 = svmla_f32_z(pred16, C20, B0, A0);
      C21 = svmla_f32_z(pred16, C21, B1, A0);

      A1  = svdup_n_f32(Aptr[baseA + 3]);
      C30 = svmla_f32_z(pred16, C30, B0, A1);
      C31 = svmla_f32_z(pred16, C31, B1, A1);

      A0  = svdup_n_f32(Aptr[baseA + 4]);
      C40 = svmla_f32_z(pred16, C40, B0, A0);
      C41 = svmla_f32_z(pred16, C41, B1, A0);

      A1  = svdup_n_f32(Aptr[baseA + 5]);
      C50 = svmla_f32_z(pred16, C50, B0, A1);
      C51 = svmla_f32_z(pred16, C51, B1, A1);

      A0  = svdup_n_f32(Aptr[baseA + 6]);
      C60 = svmla_f32_z(pred16, C60, B0, A0);
      C61 = svmla_f32_z(pred16, C61, B1, A0);
      
      A1  = svdup_n_f32(Aptr[baseA + 7]);      
      C70 = svmla_f32_z(pred16, C70, B0, A1);
      C71 = svmla_f32_z(pred16, C71, B1, A1);
      
      baseA = baseA + 10;
      baseB = baseB + 32;
    }
    
    svst1_f32(pred16, &Ctrow(0, 0),  C00);
    svst1_f32(pred16, &Ctrow(1, 0),  C10);
    svst1_f32(pred16, &Ctrow(2, 0),  C20);
    svst1_f32(pred16, &Ctrow(3, 0),  C30);
    svst1_f32(pred16, &Ctrow(4, 0),  C40);
    svst1_f32(pred16, &Ctrow(5, 0),  C50);
    svst1_f32(pred16, &Ctrow(6, 0),  C60);
    svst1_f32(pred16, &Ctrow(7, 0),  C70);
    
    svst1_f32(pred16, &Ctrow(0, 16),  C01);
    svst1_f32(pred16, &Ctrow(1, 16),  C11);
    svst1_f32(pred16, &Ctrow(2, 16),  C21);
    svst1_f32(pred16, &Ctrow(3, 16),  C31);
    svst1_f32(pred16, &Ctrow(4, 16),  C41);
    svst1_f32(pred16, &Ctrow(5, 16),  C51);
    svst1_f32(pred16, &Ctrow(6, 16),  C61);
    svst1_f32(pred16, &Ctrow(7, 16),  C71);

    for ( j=0; j<nr; j++ )
      for ( i=0; i<mr; i++ )
	Crow(i,j) = Ctrow(i, j);

  } else {

    C00 = svld1_f32(pred16, &Crow(0, 0));
    C10 = svld1_f32(pred16, &Crow(1, 0));
    C20 = svld1_f32(pred16, &Crow(2, 0));
    C30 = svld1_f32(pred16, &Crow(3, 0));
    C40 = svld1_f32(pred16, &Crow(4, 0));
    C50 = svld1_f32(pred16, &Crow(5, 0));
    C60 = svld1_f32(pred16, &Crow(6, 0));
    C70 = svld1_f32(pred16, &Crow(7, 0));
    C80 = svld1_f32(pred16, &Crow(8, 0));
    C90 = svld1_f32(pred16, &Crow(9, 0));

    C01 = svld1_f32(pred16, &Crow(0, 16));
    C11 = svld1_f32(pred16, &Crow(1, 16));
    C21 = svld1_f32(pred16, &Crow(2, 16));
    C31 = svld1_f32(pred16, &Crow(3, 16));
    C41 = svld1_f32(pred16, &Crow(4, 16));
    C51 = svld1_f32(pred16, &Crow(5, 16));
    C61 = svld1_f32(pred16, &Crow(6, 16));
    C71 = svld1_f32(pred16, &Crow(7, 16));
    C81 = svld1_f32(pred16, &Crow(8, 16));
    C91 = svld1_f32(pred16, &Crow(9, 16));

    for ( k=0; k<kc; k++ ) {
      
      B0 = svld1_f32(pred16, &Bptr[baseB]);
      B1 = svld1_f32(pred16, &Bptr[baseB + 16]);
      
      A0=svdup_n_f32(Aptr[baseA + 0]);
      C00 = svmla_f32_z(pred16, C00, B0, A0);
      C01 = svmla_f32_z(pred16, C01, B1, A0);
      
      A1=svdup_n_f32(Aptr[baseA + 1]);
      C10 = svmla_f32_z(pred16, C10, B0, A1);
      C11 = svmla_f32_z(pred16, C11, B1, A1);
      
      A0=svdup_n_f32(Aptr[baseA + 2]);
      C20 = svmla_f32_z(pred16, C20, B0, A0);
      C21 = svmla_f32_z(pred16, C21, B1, A0);
      
      A1=svdup_n_f32(Aptr[baseA + 3]);
      C30 = svmla_f32_z(pred16, C30, B0, A1);
      C31 = svmla_f32_z(pred16, C31, B1, A1);
      
      A0=svdup_n_f32(Aptr[baseA + 4]);
      C40 = svmla_f32_z(pred16, C40, B0, A0);
      C41 = svmla_f32_z(pred16, C41, B1, A0);
      
      A1=svdup_n_f32(Aptr[baseA + 5]);
      C50 = svmla_f32_z(pred16, C50, B0, A1);
      C51 = svmla_f32_z(pred16, C51, B1, A1);
      
      A0=svdup_n_f32(Aptr[baseA + 6]);
      C60 = svmla_f32_z(pred16, C60, B0, A0);
      C61 = svmla_f32_z(pred16, C61, B1, A0);
      
      A1=svdup_n_f32(Aptr[baseA + 7]);
      C70 = svmla_f32_z(pred16, C70, B0, A1);
      C71 = svmla_f32_z(pred16, C71, B1, A1);
      
      A0=svdup_n_f32(Aptr[baseA + 8]);
      C80 = svmla_f32_z(pred16, C80, B0, A0);
      C81 = svmla_f32_z(pred16, C81, B1, A0);
      
      A1=svdup_n_f32(Aptr[baseA + 9]);
      C90 = svmla_f32_z(pred16, C90, B0, A1);
      C91 = svmla_f32_z(pred16, C91, B1, A1);
      
      baseA = baseA + Amr;
      baseB = baseB + Bnr;
    }

    svst1_f32(pred16, &Ctrow(0, 0),  C00);
    svst1_f32(pred16, &Ctrow(1, 0),  C10);
    svst1_f32(pred16, &Ctrow(2, 0),  C20);
    svst1_f32(pred16, &Ctrow(3, 0),  C30);
    svst1_f32(pred16, &Ctrow(4, 0),  C40);
    svst1_f32(pred16, &Ctrow(5, 0),  C50);
    svst1_f32(pred16, &Ctrow(6, 0),  C60);
    svst1_f32(pred16, &Ctrow(7, 0),  C70);
    svst1_f32(pred16, &Ctrow(8, 0),  C80);
    svst1_f32(pred16, &Ctrow(9, 0),  C90);
    
    svst1_f32(pred16, &Ctrow(0, 16),  C01);
    svst1_f32(pred16, &Ctrow(1, 16),  C11);
    svst1_f32(pred16, &Ctrow(2, 16),  C21);
    svst1_f32(pred16, &Ctrow(3, 16),  C31);
    svst1_f32(pred16, &Ctrow(4, 16),  C41);
    svst1_f32(pred16, &Ctrow(5, 16),  C51);
    svst1_f32(pred16, &Ctrow(6, 16),  C61);
    svst1_f32(pred16, &Ctrow(7, 16),  C71);
    svst1_f32(pred16, &Ctrow(8, 16),  C81);
    svst1_f32(pred16, &Ctrow(9, 16),  C91);

    for ( j=0; j<nr; j++ )
      for ( i=0; i<mr; i++ )
	Crow(i,j) = Ctrow(i, j);

  }
    
}


/*
  uint64_t ukc           = kc / 4;
  uint64_t ukc_left      = kc % 4;
  uint64_t uldC          = ldC;
  uint64_t mr_left       = mr;
  uint64_t whilelo       = 16;

  __asm__ volatile
  (
  // input operands
  " ldr x20, %[Aaddr]                          \n\t" // A
  " ldr x21, %[Baddr]                          \n\t" // B
  " ldr x22, %[uldC]                           \n\t" // Load ldC
  " lsl x22, x22, #2                           \n\t" // Stride multiplied by 4 (sizeof(FP32))
  " ldr x23, %[ukc]                            \n\t" 
  " ldr x24, %[ukc_left]                       \n\t" 
  " ldr x25, %[whilelo]                        \n\t"
  " ldr x10, %[mraddr]                         \n\t"
  "                                            \n\t"
  " mov x0, xzr                                \n\t" //Preparing whilelelo SVE (16)
  " whilelo p0.s, xzr, x25                     \n\t"
  "                                            \n\t"
  " ld1w  z28.s, p0/z, [x21]                   \n\t" // Load B
  " ld1w  z29.s, p0/z, [x21, #1, mul vl]       \n\t" // Load B + 16
  "                                            \n\t"
  " ld1rw z20.s, p0/z, [x20]                   \n\t" // Load and Broadcast A 1/6
  " ld1rw z21.s, p0/z, [x20, 4]                \n\t" // Load and Broadcast A 2/6
  " ld1rw z22.s, p0/z, [x20, 8]                \n\t" // Load and Broadcast A 3/6
  " ld1rw z23.s, p0/z, [x20, 12]               \n\t" // Load and Broadcast A 4/6
  " ld1rw z24.s, p0/z, [x20, 16]               \n\t" // Load and Broadcast A 5/6
  " ld1rw z25.s, p0/z, [x20, 20]               \n\t" // Load and Broadcast A 6/6
  "                                            \n\t" 
  " add x21, x21, #128                         \n\t" // Update B Address = (NR(32) *FP32(4))
  " add x20, x20, #40                          \n\t" // Update A Address = (MR(6) * FP32(4))
  "                                            \n\t" 
  " ldr   x0, %[Caddr]                         \n\t" // C
  " ld1w  z0.s, p0/z,  [x0]                    \n\t" // Load C
  " ld1w  z1.s, p0/z,  [x0, #1, mul vl]        \n\t" // Load C + 16
  " add   x1, x0, x22                          \n\t" // C + ldC...
  " ld1w  z2.s, p0/z,  [x1]                    \n\t" 
  " ld1w  z3.s, p0/z,  [x1, #1, mul vl]        \n\t" 
  " add   x2, x1, x22                          \n\t" 
  " ld1w  z4.s, p0/z,  [x2]                    \n\t"  
  " ld1w  z5.s, p0/z,  [x2, #1, mul vl]        \n\t" 
  " add   x3, x2, x22                          \n\t" 
  " ld1w  z6.s, p0/z,  [x3]                    \n\t"  
  " ld1w  z7.s, p0/z,  [x3, #1, mul vl]        \n\t" 
  " add   x4, x3, x22                          \n\t" 
  " ld1w  z8.s, p0/z,  [x4]                    \n\t"  
  " ld1w  z9.s, p0/z,  [x4, #1, mul vl]        \n\t" 
  " add   x5, x4, x22                          \n\t" 
  " ld1w  z10.s, p0/z, [x5]                    \n\t"  
  " ld1w  z11.s, p0/z, [x5, #1, mul vl]        \n\t"
  "                                            \n\t"
  " cmp x23, 0                                 \n\t" // Check end of iteration count   
  BEQ(LEFT_SVE_6x32)
  "                                            \n\t"
  LABEL(LOOP_SVE_6x32)                            
  "                                            \n\t" //------------------------------
  "                                            \n\t" // [*] LOOP UNROLL x1
  "                                            \n\t" //------------------------------
  " ld1w  z30.s, p0/z, [x21]                   \n\t" // Load and Broadcast B
  " ld1w  z31.s, p0/z, [x21, #1, mul vl]       \n\t" // Load and Broadcast B + 16
  " add x21, x21, #128                         \n\t" // Update B Address = (NR(32) *FP32(4))
  "                                            \n\t"
  " fmla z0.s, p0/m, z28.s, z20.s              \n\t"
  " fmla z1.s, p0/m, z29.s, z20.s              \n\t"
  " ld1rw z20.s, p0/z, [x20]                   \n\t" 
  "                                            \n\t"
  " fmla z2.s, p0/m, z28.s, z21.s              \n\t"
  " fmla z3.s, p0/m, z29.s, z21.s              \n\t"
  " ld1rw z21.s, p0/z, [x20, 4]                \n\t" 
  "                                            \n\t"
  " fmla z4.s, p0/m, z28.s, z22.s              \n\t"
  " fmla z5.s, p0/m, z29.s, z22.s              \n\t"
  " ld1rw z22.s, p0/z, [x20, 8]                \n\t" 
  "                                            \n\t"
  " fmla z6.s, p0/m, z28.s, z23.s              \n\t"
  " fmla z7.s, p0/m, z29.s, z23.s              \n\t"
  " ld1rw z23.s, p0/z, [x20, 12]               \n\t" 
  "                                            \n\t"
  " fmla z8.s, p0/m, z28.s, z24.s              \n\t"
  " fmla z9.s, p0/m, z29.s, z24.s              \n\t"
  " ld1rw z24.s, p0/z, [x20, 16]               \n\t" 
  "                                            \n\t"
  " fmla z10.s, p0/m, z28.s, z25.s             \n\t"
  " fmla z11.s, p0/m, z29.s, z25.s             \n\t"
  " ld1rw z25.s, p0/z, [x20, 20]               \n\t" 
  "                                            \n\t" 
  "                                            \n\t" //------------------------------
  "                                            \n\t" // [*] LOOP UNROLL x2
  "                                            \n\t" //------------------------------
  " ld1w  z28.s, p0/z, [x21]                   \n\t" // Load B
  " ld1w  z29.s, p0/z, [x21, #1, mul vl]       \n\t" // Load B + 16
  " add x21, x21, #128                         \n\t" // Update B Address = (NR(32) *FP32(4))
  "                                            \n\t" 
  " fmla z0.s, p0/m, z30.s, z20.s              \n\t"
  " fmla z1.s, p0/m, z31.s, z20.s              \n\t"
  " ld1rw z20.s, p0/z, [x20, 40]               \n\t" 
  "                                            \n\t"
  " fmla z2.s, p0/m, z30.s, z21.s              \n\t"
  " fmla z3.s, p0/m, z31.s, z21.s              \n\t"
  " ld1rw z21.s, p0/z, [x20, 44]               \n\t" 
  "                                            \n\t"
  " fmla z4.s, p0/m, z30.s, z22.s              \n\t"
  " fmla z5.s, p0/m, z31.s, z22.s              \n\t"
  " ld1rw z22.s, p0/z, [x20, 48]               \n\t" 
  "                                            \n\t"
  " fmla z6.s, p0/m, z30.s, z23.s              \n\t"
  " fmla z7.s, p0/m, z31.s, z23.s              \n\t"
  " ld1rw z23.s, p0/z, [x20, 52]               \n\t" 
  "                                            \n\t"
  " fmla z8.s, p0/m, z30.s, z24.s              \n\t"
  " fmla z9.s, p0/m, z31.s, z24.s              \n\t"
  " ld1rw z24.s, p0/z, [x20, 56]               \n\t" 
  "                                            \n\t"
  " fmla z10.s, p0/m, z30.s, z25.s             \n\t"
  " fmla z11.s, p0/m, z31.s, z25.s             \n\t"
  " ld1rw z25.s, p0/z, [x20, 60]               \n\t" 
  "                                            \n\t" 
  "                                            \n\t" //------------------------------
  "                                            \n\t" // [*] LOOP UNROLL x3
  "                                            \n\t" //------------------------------
  " ld1w  z30.s, p0/z, [x21]                   \n\t" // Load and Broadcast B
  " ld1w  z31.s, p0/z, [x21, #1, mul vl]       \n\t" // Load and Broadcast B + 16
  " add x21, x21, #128                         \n\t" // Update B Address = (NR(32) *FP32(4))
  "                                            \n\t"
  " fmla z0.s, p0/m, z28.s, z20.s              \n\t"
  " fmla z1.s, p0/m, z29.s, z20.s              \n\t"
  " ld1rw z20.s, p0/z, [x20, 80]               \n\t" 
  "                                            \n\t"
  " fmla z2.s, p0/m, z28.s, z21.s              \n\t"
  " fmla z3.s, p0/m, z29.s, z21.s              \n\t"
  " ld1rw z21.s, p0/z, [x20, 84]               \n\t" 
  "                                            \n\t"
  " fmla z4.s, p0/m, z28.s, z22.s              \n\t"
  " fmla z5.s, p0/m, z29.s, z22.s              \n\t"
  " ld1rw z22.s, p0/z, [x20, 88]               \n\t" 
  "                                            \n\t"
  " fmla z6.s, p0/m, z28.s, z23.s              \n\t"
  " fmla z7.s, p0/m, z29.s, z23.s              \n\t"
  " ld1rw z23.s, p0/z, [x20, 92]               \n\t" 
  "                                            \n\t"
  " fmla z8.s, p0/m, z28.s, z24.s              \n\t"
  " fmla z9.s, p0/m, z29.s, z24.s              \n\t"
  " ld1rw z24.s, p0/z, [x20, 96]               \n\t" 
  "                                            \n\t"
  " fmla z10.s, p0/m, z28.s, z25.s             \n\t"
  " fmla z11.s, p0/m, z29.s, z25.s             \n\t"
  " ld1rw z25.s, p0/z, [x20, 100]               \n\t" 
  "                                            \n\t" 
  "                                            \n\t" //------------------------------
  "                                            \n\t" // [*] LOOP UNROLL x4
  "                                            \n\t" //------------------------------
  " ld1w  z28.s, p0/z, [x21]                   \n\t" // Load B
  " ld1w  z29.s, p0/z, [x21, #1, mul vl]       \n\t" // Load B + 16
  " add x21, x21, #128                         \n\t" // Update B Address = (NR(32) *FP32(4))
  "                                            \n\t" 
  " fmla z0.s, p0/m, z30.s, z20.s              \n\t"
  " fmla z1.s, p0/m, z31.s, z20.s              \n\t"
  " ld1rw z20.s, p0/z, [x20, 120]               \n\t" 
  "                                            \n\t"
  " fmla z2.s, p0/m, z30.s, z21.s              \n\t"
  " fmla z3.s, p0/m, z31.s, z21.s              \n\t"
  " ld1rw z21.s, p0/z, [x20, 124]               \n\t" 
  "                                            \n\t"
  " fmla z4.s, p0/m, z30.s, z22.s              \n\t"
  " fmla z5.s, p0/m, z31.s, z22.s              \n\t"
  " ld1rw z22.s, p0/z, [x20, 128]               \n\t" 
  "                                            \n\t"
  " fmla z6.s, p0/m, z30.s, z23.s              \n\t"
  " fmla z7.s, p0/m, z31.s, z23.s              \n\t"
  " ld1rw z23.s, p0/z, [x20, 132]               \n\t" 
  "                                            \n\t"
  " fmla z8.s, p0/m, z30.s, z24.s              \n\t"
  " fmla z9.s, p0/m, z31.s, z24.s              \n\t"
  " ld1rw z24.s, p0/z, [x20, 136]               \n\t" 
  "                                            \n\t"
  " fmla z10.s, p0/m, z30.s, z25.s             \n\t"
  " fmla z11.s, p0/m, z31.s, z25.s             \n\t"
  " ld1rw z25.s, p0/z, [x20, 140]               \n\t" 
  "                                            \n\t"
  " add x20, x20, #160                         \n\t" 
  "                                            \n\t"
  " sub x23, x23, 1                            \n\t" // Decrease iteration count by 1
  " cmp x23, 0                                 \n\t" // Check end of iteration count   
  BNE(LOOP_SVE_6x32)
  "                                            \n\t"
  LABEL(LEFT_SVE_6x32)
  "                                            \n\t"
  " cmp x24, 0                                 \n\t" // Check end of iteration count   
  BEQ(STORE_SVE_6x32)
  "                                            \n\t" //------------------------------
  "                                            \n\t" // [*] K LEFT, (x1)
  "                                            \n\t" //------------------------------
  " fmla z0.s, p0/m, z28.s, z20.s              \n\t"
  " fmla z1.s, p0/m, z29.s, z20.s              \n\t"
  "                                            \n\t"
  " fmla z2.s, p0/m, z28.s, z21.s              \n\t"
  " fmla z3.s, p0/m, z29.s, z21.s              \n\t"
  "                                            \n\t"
  " fmla z4.s, p0/m, z28.s, z22.s              \n\t"
  " fmla z5.s, p0/m, z29.s, z22.s              \n\t"
  "                                            \n\t"
  " fmla z6.s, p0/m, z28.s, z23.s              \n\t"
  " fmla z7.s, p0/m, z29.s, z23.s              \n\t"
  "                                            \n\t"
  " fmla z8.s, p0/m, z28.s, z24.s              \n\t"
  " fmla z9.s, p0/m, z29.s, z24.s              \n\t"
  "                                            \n\t"
  " fmla z10.s, p0/m, z28.s, z25.s             \n\t"
  " fmla z11.s, p0/m, z29.s, z25.s             \n\t"
  "                                            \n\t"
  " sub x24, x24, 1                            \n\t"
  " cmp x24, 0                                 \n\t" 
  BEQ(STORE_SVE_6x32)
  "                                            \n\t" //------------------------------
  "                                            \n\t" // [*] K LEFT, (x2)
  "                                            \n\t" //------------------------------
  " add x20, x20, #40                          \n\t" // Update A Address = (MR(6)  * FP32(4))
  " add x21, x21, #128                         \n\t" // Update B Address = (NR(32) * FP32(4))  
  "                                            \n\t"
  " ld1w  z28.s, p0/z, [x21]                   \n\t" 
  " ld1w  z29.s, p0/z, [x21, #1, mul vl]       \n\t" 
  "                                            \n\t"
  " ld1rw z20.s, p0/z, [x20]                   \n\t" 
  " ld1rw z21.s, p0/z, [x20, 4]                \n\t" 
  " ld1rw z22.s, p0/z, [x20, 8]                \n\t" 
  " ld1rw z23.s, p0/z, [x20, 12]               \n\t" 
  " ld1rw z24.s, p0/z, [x20, 16]               \n\t" 
  " ld1rw z25.s, p0/z, [x20, 20]               \n\t" 
  "                                            \n\t" 
  " fmla z0.s, p0/m, z28.s, z20.s              \n\t"
  " fmla z1.s, p0/m, z29.s, z20.s              \n\t"
  "                                            \n\t"
  " fmla z2.s, p0/m, z28.s, z21.s              \n\t"
  " fmla z3.s, p0/m, z29.s, z21.s              \n\t"
  "                                            \n\t"
  " fmla z4.s, p0/m, z28.s, z22.s              \n\t"
  " fmla z5.s, p0/m, z29.s, z22.s              \n\t"
  "                                            \n\t"
  " fmla z6.s, p0/m, z28.s, z23.s              \n\t"
  " fmla z7.s, p0/m, z29.s, z23.s              \n\t"
  "                                            \n\t"
  " fmla z8.s, p0/m, z28.s, z24.s              \n\t"
  " fmla z9.s, p0/m, z29.s, z24.s              \n\t"
  "                                            \n\t"
  " fmla z10.s, p0/m, z28.s, z25.s             \n\t"
  " fmla z11.s, p0/m, z29.s, z25.s             \n\t"
  "                                            \n\t"
  " sub x24, x24, 1                            \n\t"
  " cmp x24, 0                                 \n\t"
  BEQ(STORE_SVE_6x32)
  "                                            \n\t" //------------------------------
  "                                            \n\t" // [*] K LEFT, (x3) <<Last>>
  "                                            \n\t" //------------------------------
  " add x20, x20, #40                          \n\t" // Update A Address = (MR(10) * FP32(4))
  " add x21, x21, #128                         \n\t" // Update B Address = (NR(32) * FP32(4))  
  "                                            \n\t"
  " ld1w  z28.s, p0/z, [x21]                   \n\t" 
  " ld1w  z29.s, p0/z, [x21, #1, mul vl]       \n\t" 
  "                                            \n\t"
  " ld1rw z20.s, p0/z, [x20]                   \n\t" 
  " ld1rw z21.s, p0/z, [x20, 4]                \n\t" 
  " ld1rw z22.s, p0/z, [x20, 8]                \n\t" 
  " ld1rw z23.s, p0/z, [x20, 12]               \n\t" 
  " ld1rw z24.s, p0/z, [x20, 16]               \n\t" 
  " ld1rw z25.s, p0/z, [x20, 20]               \n\t" 
  "                                            \n\t" 
  " fmla z0.s, p0/m, z28.s, z20.s              \n\t"
  " fmla z1.s, p0/m, z29.s, z20.s              \n\t"
  "                                            \n\t"
  " fmla z2.s, p0/m, z28.s, z21.s              \n\t"
  " fmla z3.s, p0/m, z29.s, z21.s              \n\t"
  "                                            \n\t"
  " fmla z4.s, p0/m, z28.s, z22.s              \n\t"
  " fmla z5.s, p0/m, z29.s, z22.s              \n\t"
  "                                            \n\t"
  " fmla z6.s, p0/m, z28.s, z23.s              \n\t"
  " fmla z7.s, p0/m, z29.s, z23.s              \n\t"
  "                                            \n\t"
  " fmla z8.s, p0/m, z28.s, z24.s              \n\t"
  " fmla z9.s, p0/m, z29.s, z24.s              \n\t"
  "                                            \n\t"
  " fmla z10.s, p0/m, z28.s, z25.s             \n\t"
  " fmla z11.s, p0/m, z29.s, z25.s             \n\t"
  "                                            \n\t"
  LABEL(STORE_SVE_6x32)
  " st1w z0.s,  p0, [x0]                       \n\t" // Store C x0
  " st1w z1.s,  p0, [x0, #1, mul vl]           \n\t" // Store C x0 + 16
  " sub x10, x10, 1                            \n\t"
  "                                            \n\t"
  " st1w z2.s,  p0, [x1]                       \n\t" 
  " st1w z3.s,  p0, [x1, #1, mul vl]           \n\t" 
  " sub x10, x10, 1                            \n\t"
  "                                            \n\t"
  " st1w z4.s,  p0, [x2]                       \n\t" 
  " st1w z5.s,  p0, [x2, #1, mul vl]           \n\t"
  " sub x10, x10, 1                            \n\t"
  "                                            \n\t"
  " st1w z6.s,  p0, [x3]                       \n\t" 
  " st1w z7.s,  p0, [x3, #1, mul vl]           \n\t" 
  " sub x10, x10, 1                            \n\t"
  " cmp x10, 0                                 \n\t" 
  BEQ(END_6x32)
  "                                            \n\t"
  " st1w z8.s,  p0, [x4]                       \n\t" 
  " st1w z9.s,  p0, [x4, #1, mul vl]           \n\t"
  "                                            \n\t"
  " sub x10, x10, 1                            \n\t"
  " cmp x10, 0                                 \n\t" 
  BEQ(END_6x32)
  "                                            \n\t"
  " st1w z10.s, p0, [x5]                       \n\t" 
  " st1w z11.s, p0, [x5, #1, mul vl]           \n\t" 
  "                                            \n\t"
  LABEL(END_6x32)
  : // output operands (none)
  : // input operands
  [ukc]      "m" (ukc),     
  [ukc_left] "m" (ukc_left),
  [Aaddr]    "m" (Ar),      
  [Baddr]    "m" (Br),
  [Caddr]    "m" (C),
  [uldC]     "m" (uldC),
  [mraddr]   "m" (mr_left),
  [whilelo]  "m" (whilelo) 
  : // Memory Dir. Register 
  "x0",  "x1",  "x2",  "x3",  "x4", 
  "x5",  "x6",  "x7",  "x8",  "x9",
  "x20","x21", "x22", "x23", "x24",
  "x25","x10", "x12",
  // SVE Register 
  "z0",  "z1",  "z2",  "z3",
  "z4",  "z5",  "z6",  "z7",
      "z8",  "z9", "z10", "z11",
      "z12", "z13", "z14", "z15",
      "z16", "z17", "z18", "z19",
      "z20", "z21", "z22", "z23",
      "z24", "z25", "z26", "z27",
      "z28", "z29", "z30", "z31");
    */    

//=============================================================================================//
//====================== 10 X 32    A S M    M I C R O - K E R N E L ===========================//
//=============================================================================================//




void gemm_microkernel_neon_sve_fixed_4x16_fp32( int mr, int nr, int kc, float alpha, float *Ar, float *Br, float beta, float *C, int ldC ) {
/*
  BLIS GEMM microkernel, computes the product Cr := Cr + Ar * Br
  Specific: only for MRxNR = 4x16
*/
  //SET_MR_NR(4, 16);
  
  int         i, j, k, baseB = 0, baseA = 0, ldCt = NR, Amr, Bnr;
  svfloat32_t C00, C10, C20, C30, B0, B1, B2, B3, A0, C0n, C1n, C2n, C3n;  
  
  float   zero = 0.0, one = 1.0, *Aptr, *Bptr, Ctmp[MR * NR];

  svbool_t pred16 = svwhilelt_b32_u32(0, 16);
  //svbool_t pred4  = svwhilelt_b32_u32(0, 4);
  
  
  if ( kc==0 )
    return;
  
  //Set to 0
  //Set to 0
  C00=svdup_n_f32(0); //C01=svdup_n_f32(0); C02=svdup_n_f32(0); C03=svdup_n_f32(0);
  C10=svdup_n_f32(0); //C11=svdup_n_f32(0); C12=svdup_n_f32(0); C13=svdup_n_f32(0);
  C20=svdup_n_f32(0); //C21=svdup_n_f32(0); C22=svdup_n_f32(0); C23=svdup_n_f32(0);
  C30=svdup_n_f32(0); //C31=svdup_n_f32(0); C32=svdup_n_f32(0); C33=svdup_n_f32(0);

  Aptr = &Ar[0];
  Amr  = MR;
  Bptr = &Br[0];
  Bnr  = NR;

  if ( alpha!=zero ) {
    for ( k=0; k<kc; k++ ) {
      // Load colums/rows of A/B for current iteration
      //A0 = svld1_f32(pred16, &Aptr[baseA]);
      B0 = svld1_f32(pred16, &Bptr[baseB]);
      
      A0=svdup_n_f32(Aptr[baseA + 0]);
      C00 = svmla_f32_z(pred16, C00, B0, A0);
      
      A0=svdup_n_f32(Aptr[baseA + 1]);
      C10 = svmla_f32_z(pred16, C10, B0, A0);
      
      A0=svdup_n_f32(Aptr[baseA + 2]);
      C20 = svmla_f32_z(pred16, C20, B0, A0);
      
      A0=svdup_n_f32(Aptr[baseA + 3]);
      C30 = svmla_f32_z(pred16, C30, B0, A0);
      
      baseA = baseA + Amr;
      baseB = baseB + Bnr;
    }
    
    if ( alpha==-one ) {
      C00 = svmul_n_f32_x(pred16, C00, -1.0);
      C10 = svmul_n_f32_x(pred16, C10, -1.0);
      C20 = svmul_n_f32_x(pred16, C20, -1.0);
      C30 = svmul_n_f32_x(pred16, C30, -1.0);

    }
    else if ( alpha!=one ) {
      C00 = svmul_n_f32_x(pred16, C00, alpha);
      C10 = svmul_n_f32_x(pred16, C10, alpha);
      C20 = svmul_n_f32_x(pred16, C20, alpha);
      C30 = svmul_n_f32_x(pred16, C30, alpha);
    }
  }
  
  if ( (mr<MR) || (nr<NR) ) {

    svst1_f32(pred16, &Ctrow(0, 0),  C00);    
    svst1_f32(pred16, &Ctrow(1, 0),  C10);    
    svst1_f32(pred16, &Ctrow(2, 0),  C20);    
    svst1_f32(pred16, &Ctrow(3, 0),  C30);

    //printf("----------> (j=%d < nr=%d) - (i=%d < mr=%d), MR=%d, NR=%d\n", j, nr, i, mr, MR, NR);
    if ( beta!=zero ) {
      for ( j=0; j<nr; j++ )
        for ( i=0; i<mr; i++ ) {
	  //printf("C[%d, %d] = %0.5f\n", i, j, Ctrow(i, j));
          Crow(i,j) = beta*Crow(i,j) + Ctrow(i,j);
	}
    }
    
    else {
      for ( j=0; j<nr; j++ )
        for ( i=0; i<mr; i++ )
          Crow(i,j) = Ctrow(i,j);
    }
  }
  else if ( (mr==MR) && (nr==NR) ) {
    
    if (beta != zero) {
      C0n = svld1_f32(pred16, &Crow(0,0));
      C1n = svld1_f32(pred16, &Crow(1,0));
      C2n = svld1_f32(pred16, &Crow(2,0));
      C3n = svld1_f32(pred16, &Crow(3,0));
      
      //TODO: BETA MULTIPY
      C00 = svadd_f32_z(pred16, C0n, C00);
      C10 = svadd_f32_z(pred16, C1n, C10);
      C20 = svadd_f32_z(pred16, C2n, C20);
      C30 = svadd_f32_z(pred16, C3n, C30);
      
    }
    
    svst1_f32(pred16, &Crow(0, 0), C00);
    svst1_f32(pred16, &Crow(1, 0), C10);
    svst1_f32(pred16, &Crow(2, 0), C20);
    svst1_f32(pred16, &Crow(3, 0), C30);

  }
  else {
    printf("Error: Incorrect use of 4x16 micro-kernel with %d x %d block\n", mr, nr);
    exit(-1);
  }
}

void gemm_microkernel_neon_sve_fixed_4x32_fp32( int mr, int nr, int kc, float alpha, float *Ar, float *Br, float beta, float *C, int ldC ) {
/*
  BLIS GEMM microkernel, computes the product Cr := Cr + Ar * Br
  Specific: only for MRxNR = 4x16
*/
  //SET_MR_NR(4, 32);
  
  int         i, j, k, baseB = 0, baseA = 0, ldCt = NR, Amr, Bnr;
  svfloat32_t C00, C10, C20, C30,
              C01, C11, C21, C31,
              B0,  B1,  B2,  B3, A0,
              C0n, C1n, C2n, C3n, Bta;  
  
  float   zero = 0.0, one = 1.0, *Aptr, *Bptr, Ctmp[MR * NR];

  svbool_t pred16 = svwhilelt_b32_u32(0, 16);
  
  
  if ( kc==0 )
    return;
  

  //Set Beta
  Bta=svdup_n_f32(beta);
  //Set to 0
  C00=svdup_n_f32(0); C01=svdup_n_f32(0); //C02=svdup_n_f32(0); C03=svdup_n_f32(0);
  C10=svdup_n_f32(0); C11=svdup_n_f32(0); //C12=svdup_n_f32(0); C13=svdup_n_f32(0);
  C20=svdup_n_f32(0); C21=svdup_n_f32(0); //C22=svdup_n_f32(0); C23=svdup_n_f32(0);
  C30=svdup_n_f32(0); C31=svdup_n_f32(0); //C32=svdup_n_f32(0); C33=svdup_n_f32(0);

  Aptr = &Ar[0];
  Amr  = MR;
  Bptr = &Br[0];
  Bnr  = NR;

  if ( alpha!=zero ) {
    for ( k=0; k<kc; k++ ) {
      // Load colums/rows of A/B for current iteration
      //A0 = svld1_f32(pred16, &Aptr[baseA]);
      B0 = svld1_f32(pred16, &Bptr[baseB]);
      B1 = svld1_f32(pred16, &Bptr[baseB + 16]);

      
      A0=svdup_n_f32(Aptr[baseA + 0]);
      C00 = svmla_f32_z(pred16, C00, B0, A0);
      C01 = svmla_f32_z(pred16, C01, B1, A0);
      
      A0=svdup_n_f32(Aptr[baseA + 1]);
      C10 = svmla_f32_z(pred16, C10, B0, A0);
      C11 = svmla_f32_z(pred16, C11, B1, A0);
      
      A0=svdup_n_f32(Aptr[baseA + 2]);
      C20 = svmla_f32_z(pred16, C20, B0, A0);
      C21 = svmla_f32_z(pred16, C21, B1, A0);
      
      A0=svdup_n_f32(Aptr[baseA + 3]);
      C30 = svmla_f32_z(pred16, C30, B0, A0);
      C31 = svmla_f32_z(pred16, C31, B1, A0);
      
      baseA = baseA + Amr;
      baseB = baseB + Bnr;
    }
    
    if ( alpha==-one ) {
      C00 = svmul_n_f32_x(pred16, C00, -1.0);
      C10 = svmul_n_f32_x(pred16, C10, -1.0);
      C20 = svmul_n_f32_x(pred16, C20, -1.0);
      C30 = svmul_n_f32_x(pred16, C30, -1.0);
      
      C01 = svmul_n_f32_x(pred16, C01, -1.0);
      C11 = svmul_n_f32_x(pred16, C11, -1.0);
      C21 = svmul_n_f32_x(pred16, C21, -1.0);
      C31 = svmul_n_f32_x(pred16, C31, -1.0);
    } else if ( alpha!=one ) {
      C00 = svmul_n_f32_x(pred16, C00, alpha);
      C10 = svmul_n_f32_x(pred16, C10, alpha);
      C20 = svmul_n_f32_x(pred16, C20, alpha);
      C30 = svmul_n_f32_x(pred16, C30, alpha);
      
      C01 = svmul_n_f32_x(pred16, C01, alpha);
      C11 = svmul_n_f32_x(pred16, C11, alpha);
      C21 = svmul_n_f32_x(pred16, C21, alpha);
      C31 = svmul_n_f32_x(pred16, C31, alpha);
    }
  }
  
  if ( (mr<MR) || (nr<NR) ) {
    svst1_f32(pred16, &Ctrow(0, 0),  C00);    
    svst1_f32(pred16, &Ctrow(1, 0),  C10);    
    svst1_f32(pred16, &Ctrow(2, 0),  C20);    
    svst1_f32(pred16, &Ctrow(3, 0),  C30);
    
    svst1_f32(pred16, &Ctrow(0, 16),  C01);    
    svst1_f32(pred16, &Ctrow(1, 16),  C11);    
    svst1_f32(pred16, &Ctrow(2, 16),  C21);    
    svst1_f32(pred16, &Ctrow(3, 16),  C31);
    
    if ( beta!=zero ) {
      for ( j=0; j<nr; j++ )
        for ( i=0; i<mr; i++ ) {
	  //printf("C[%d, %d] = %0.5f\n", i, j, Ctrow(i, j));
          Crow(i,j) = beta*Crow(i,j) + Ctrow(i,j);
	}
    }
    
    else {
      for ( j=0; j<nr; j++ )
        for ( i=0; i<mr; i++ )
          Crow(i,j) = Ctrow(i,j);
    }
  }
  else if ( (mr==MR) && (nr==NR) ) {

    if (beta != zero) {
      C0n = svld1_f32(pred16, &Crow(0,0));
      C1n = svld1_f32(pred16, &Crow(1,0));
      C2n = svld1_f32(pred16, &Crow(2,0));
      C3n = svld1_f32(pred16, &Crow(3,0));
      
      C00 = svmla_f32_z(pred16, C00, C0n, Bta);
      C10 = svmla_f32_z(pred16, C10, C1n, Bta);
      C20 = svmla_f32_z(pred16, C20, C2n, Bta);
      C30 = svmla_f32_z(pred16, C30, C3n, Bta);
      
      //C00 = svadd_f32_z(pred16, C0n, C00);
      //C10 = svadd_f32_z(pred16, C1n, C10);
      //C20 = svadd_f32_z(pred16, C2n, C20);
      //C30 = svadd_f32_z(pred16, C3n, C30);
      
      C0n = svld1_f32(pred16, &Crow(0,16));
      C1n = svld1_f32(pred16, &Crow(1,16));
      C2n = svld1_f32(pred16, &Crow(2,16));
      C3n = svld1_f32(pred16, &Crow(3,16));

      C01 = svmla_f32_z(pred16, C01, C0n, Bta);
      C11 = svmla_f32_z(pred16, C11, C1n, Bta);
      C21 = svmla_f32_z(pred16, C21, C2n, Bta);
      C31 = svmla_f32_z(pred16, C31, C3n, Bta);

      //C01 = svadd_f32_z(pred16, C0n, C01);
      //C11 = svadd_f32_z(pred16, C1n, C11);
      //C21 = svadd_f32_z(pred16, C2n, C21);
      //C31 = svadd_f32_z(pred16, C3n, C31);      
    }
    
    svst1_f32(pred16, &Crow(0, 0), C00);
    svst1_f32(pred16, &Crow(1, 0), C10);
    svst1_f32(pred16, &Crow(2, 0), C20);
    svst1_f32(pred16, &Crow(3, 0), C30);

    svst1_f32(pred16, &Crow(0, 16), C01);
    svst1_f32(pred16, &Crow(1, 16), C11);
    svst1_f32(pred16, &Crow(2, 16), C21);
    svst1_f32(pred16, &Crow(3, 16), C31);

  }
  else {
    printf("Error: Incorrect use of 4x16 micro-kernel with %d x %d block\n", mr, nr);
    exit(-1);
  }
}

/* void gemm_microkernel_neon_sve_10x32_fp32( int mr, int nr, int kc, float alpha, float *Ar, float *Br, float beta, float *C, int ldC ) { */
/* /\* */
/*   BLIS GEMM microkernel, computes the product Cr := Cr + Ar * Br */
/*   Specific: only for MRxNR = 4x16 */
/* *\/ */
/*   SET_MR_NR(10, 32); */
  
/*   int         i, j, k, baseB = 0, baseA = 0, ldCt = NR, Amr, Bnr; */

/*   //Registers for C = (20) */
/*   svfloat32_t C00, C10, C20, C30, C40, C50, C60, C70, C80, C90, */
/*               C01, C11, C21, C31, C41, C51, C61, C71, C81, C91; */

/*   //Other auxiliar Registers (11) */
/*   svfloat32_t  A0, B0, B1, Bta, C0n, C1n, C2n, C3n, C4n, C5n, C6n, C7n, C8n, C9n; */

/*   //  #define A0 C7n */
/*   //#define B0 C8n */
/*   //#define B1 C9n */

/*   float   zero = 0.0, one = 1.0, *Aptr, *Bptr, Ctmp[MR * NR]; */
/*   svbool_t pred16 = svwhilelt_b32_u32(0, 16); */
  
/*   if ( kc==0 ) */
/*     return; */
    
/*   //Set to 0 */
/*   C00=svdup_n_f32(0); C01=svdup_n_f32(0);  */
/*   C10=svdup_n_f32(0); C11=svdup_n_f32(0);  */
/*   C20=svdup_n_f32(0); C21=svdup_n_f32(0);  */
/*   C30=svdup_n_f32(0); C31=svdup_n_f32(0);  */
/*   C40=svdup_n_f32(0); C41=svdup_n_f32(0);  */
/*   C50=svdup_n_f32(0); C51=svdup_n_f32(0);  */
/*   C60=svdup_n_f32(0); C61=svdup_n_f32(0);  */
/*   C70=svdup_n_f32(0); C71=svdup_n_f32(0); */
/*   C80=svdup_n_f32(0); C81=svdup_n_f32(0); */
/*   C90=svdup_n_f32(0); C91=svdup_n_f32(0);  */

/*   Aptr = &Ar[0]; */
/*   Amr  = MR; */
/*   Bptr = &Br[0]; */
/*   Bnr  = NR; */
  
/*   if ((mr<=5) && (nr<=16)) { */
/*     for ( k=0; k<kc; k++ ) { */
/*       B0 = svld1_f32(pred16, &Bptr[baseB]); */
      
/*       A0  = svdup_n_f32(Aptr[baseA]);       */
/*       C00 = svmla_f32_z(pred16, C00, B0, A0); */
      
/*       A0  = svdup_n_f32(Aptr[baseA + 1]);       */
/*       C10 = svmla_f32_z(pred16, C10, B0, A0); */
      
/*       A0  = svdup_n_f32(Aptr[baseA + 2]);       */
/*       C20 = svmla_f32_z(pred16, C20, B0, A0); */
      
/*       A0  = svdup_n_f32(Aptr[baseA + 3]);       */
/*       C30 = svmla_f32_z(pred16, C30, B0, A0); */
      
/*       A0  = svdup_n_f32(Aptr[baseA + 4]);       */
/*       C40 = svmla_f32_z(pred16, C40, B0, A0); */
      
/*       baseA = baseA + Amr; */
/*       baseB = baseB + Bnr; */
/*     }   */
/*   } else if ((mr<=10) && (nr<=16)) { */
/*     for ( k=0; k<kc; k++ ) { */
/*       B0 = svld1_f32(pred16, &Bptr[baseB]); */
      
/*       A0  = svdup_n_f32(Aptr[baseA]);       */
/*       C00 = svmla_f32_z(pred16, C00, B0, A0); */
      
/*       A0  = svdup_n_f32(Aptr[baseA + 1]);       */
/*       C10 = svmla_f32_z(pred16, C10, B0, A0); */
      
/*       A0  = svdup_n_f32(Aptr[baseA + 2]);       */
/*       C20 = svmla_f32_z(pred16, C20, B0, A0); */
      
/*       A0  = svdup_n_f32(Aptr[baseA + 3]);       */
/*       C30 = svmla_f32_z(pred16, C30, B0, A0); */
      
/*       A0  = svdup_n_f32(Aptr[baseA + 4]);       */
/*       C40 = svmla_f32_z(pred16, C40, B0, A0); */

/*       A0  = svdup_n_f32(Aptr[baseA + 5]);       */
/*       C50 = svmla_f32_z(pred16, C50, B0, A0); */

/*       A0  = svdup_n_f32(Aptr[baseA + 6]);       */
/*       C60 = svmla_f32_z(pred16, C60, B0, A0); */

/*       A0  = svdup_n_f32(Aptr[baseA + 7]);       */
/*       C70 = svmla_f32_z(pred16, C70, B0, A0); */

/*       A0  = svdup_n_f32(Aptr[baseA + 8]);       */
/*       C80 = svmla_f32_z(pred16, C80, B0, A0); */

/*       A0  = svdup_n_f32(Aptr[baseA + 9]);       */
/*       C90 = svmla_f32_z(pred16, C90, B0, A0); */
      
/*       baseA = baseA + Amr; */
/*       baseB = baseB + Bnr; */
/*     } */
/*   } else */
/*     for ( k=0; k<kc; k++ ) { */
/*       // Load colums/rows of A/B for current iteration */
/*       B0 = svld1_f32(pred16, &Bptr[baseB]); */
/*       B1 = svld1_f32(pred16, &Bptr[baseB + 16]); */
      
/*       A0=svdup_n_f32(Aptr[baseA + 0]); */
/*       C00 = svmla_f32_z(pred16, C00, B0, A0); */
/*       C01 = svmla_f32_z(pred16, C01, B1, A0); */
      
/*       A0=svdup_n_f32(Aptr[baseA + 1]); */
/*       C10 = svmla_f32_z(pred16, C10, B0, A0); */
/*       C11 = svmla_f32_z(pred16, C11, B1, A0); */
      
/*       A0=svdup_n_f32(Aptr[baseA + 2]); */
/*       C20 = svmla_f32_z(pred16, C20, B0, A0); */
/*       C21 = svmla_f32_z(pred16, C21, B1, A0); */
      
/*       A0=svdup_n_f32(Aptr[baseA + 3]); */
/*       C30 = svmla_f32_z(pred16, C30, B0, A0); */
/*       C31 = svmla_f32_z(pred16, C31, B1, A0); */
      
/*       A0=svdup_n_f32(Aptr[baseA + 4]); */
/*       C40 = svmla_f32_z(pred16, C40, B0, A0); */
/*       C41 = svmla_f32_z(pred16, C41, B1, A0); */
      
/*       A0=svdup_n_f32(Aptr[baseA + 5]); */
/*       C50 = svmla_f32_z(pred16, C50, B0, A0); */
/*       C51 = svmla_f32_z(pred16, C51, B1, A0); */
      
/*       A0=svdup_n_f32(Aptr[baseA + 6]); */
/*       C60 = svmla_f32_z(pred16, C60, B0, A0); */
/*       C61 = svmla_f32_z(pred16, C61, B1, A0); */
      
/*       A0=svdup_n_f32(Aptr[baseA + 7]); */
/*       C70 = svmla_f32_z(pred16, C70, B0, A0); */
/*       C71 = svmla_f32_z(pred16, C71, B1, A0); */
      
/*       A0=svdup_n_f32(Aptr[baseA + 8]); */
/*       C80 = svmla_f32_z(pred16, C80, B0, A0); */
/*       C81 = svmla_f32_z(pred16, C81, B1, A0); */
      
/*       A0=svdup_n_f32(Aptr[baseA + 9]); */
/*       C90 = svmla_f32_z(pred16, C90, B0, A0); */
/*       C91 = svmla_f32_z(pred16, C91, B1, A0); */
      
/*       baseA = baseA + Amr; */
/*       baseB = baseB + Bnr; */
/*     } */
  
/*   if ( alpha==-one ) { */
/*     C00 = svmul_n_f32_x(pred16, C00, -1.0); */
/*     C10 = svmul_n_f32_x(pred16, C10, -1.0); */
/*     C20 = svmul_n_f32_x(pred16, C20, -1.0); */
/*     C30 = svmul_n_f32_x(pred16, C30, -1.0); */
/*     C40 = svmul_n_f32_x(pred16, C40, -1.0); */
/*     C50 = svmul_n_f32_x(pred16, C50, -1.0); */
/*     C60 = svmul_n_f32_x(pred16, C60, -1.0); */
/*     C70 = svmul_n_f32_x(pred16, C70, -1.0); */
/*     C80 = svmul_n_f32_x(pred16, C80, -1.0); */
/*     C90 = svmul_n_f32_x(pred16, C90, -1.0); */
    
/*     C01 = svmul_n_f32_x(pred16, C01, -1.0); */
/*     C11 = svmul_n_f32_x(pred16, C11, -1.0); */
/*     C21 = svmul_n_f32_x(pred16, C21, -1.0); */
/*     C31 = svmul_n_f32_x(pred16, C31, -1.0); */
/*     C41 = svmul_n_f32_x(pred16, C40, -1.0); */
/*     C51 = svmul_n_f32_x(pred16, C51, -1.0); */
/*     C61 = svmul_n_f32_x(pred16, C61, -1.0); */
/*     C71 = svmul_n_f32_x(pred16, C71, -1.0); */
/*     C81 = svmul_n_f32_x(pred16, C81, -1.0); */
/*     C91 = svmul_n_f32_x(pred16, C91, -1.0); */
/*   } else if ( alpha!=one ) { */
/*     C00 = svmul_n_f32_x(pred16, C00, alpha); */
/*     C10 = svmul_n_f32_x(pred16, C10, alpha); */
/*     C20 = svmul_n_f32_x(pred16, C20, alpha); */
/*     C30 = svmul_n_f32_x(pred16, C30, alpha); */
/*     C40 = svmul_n_f32_x(pred16, C40, alpha); */
/*     C50 = svmul_n_f32_x(pred16, C50, alpha); */
/*     C60 = svmul_n_f32_x(pred16, C60, alpha); */
/*     C70 = svmul_n_f32_x(pred16, C70, alpha); */
/*     C80 = svmul_n_f32_x(pred16, C80, alpha); */
/*     C90 = svmul_n_f32_x(pred16, C90, alpha); */
    
/*     C01 = svmul_n_f32_x(pred16, C01, alpha); */
/*     C11 = svmul_n_f32_x(pred16, C11, alpha); */
/*     C21 = svmul_n_f32_x(pred16, C21, alpha); */
/*     C31 = svmul_n_f32_x(pred16, C31, alpha); */
/*     C41 = svmul_n_f32_x(pred16, C41, alpha); */
/*     C51 = svmul_n_f32_x(pred16, C51, alpha); */
/*     C61 = svmul_n_f32_x(pred16, C61, alpha); */
/*     C71 = svmul_n_f32_x(pred16, C71, alpha); */
/*     C81 = svmul_n_f32_x(pred16, C81, alpha); */
/*     C91 = svmul_n_f32_x(pred16, C91, alpha); */
/*   }   */
  
/*   if ( (mr<MR) || (nr<NR) ) { */
/*     svst1_f32(pred16, &Ctrow(0, 0),  C00);     */
/*     svst1_f32(pred16, &Ctrow(1, 0),  C10);     */
/*     svst1_f32(pred16, &Ctrow(2, 0),  C20);     */
/*     svst1_f32(pred16, &Ctrow(3, 0),  C30); */
/*     svst1_f32(pred16, &Ctrow(4, 0),  C40);     */
/*     svst1_f32(pred16, &Ctrow(5, 0),  C50);     */
/*     svst1_f32(pred16, &Ctrow(6, 0),  C60);     */
/*     svst1_f32(pred16, &Ctrow(7, 0),  C70); */
/*     svst1_f32(pred16, &Ctrow(8, 0),  C80); */
/*     svst1_f32(pred16, &Ctrow(9, 0),  C90); */
    
/*     svst1_f32(pred16, &Ctrow(0, 16),  C01);     */
/*     svst1_f32(pred16, &Ctrow(1, 16),  C11);     */
/*     svst1_f32(pred16, &Ctrow(2, 16),  C21);     */
/*     svst1_f32(pred16, &Ctrow(3, 16),  C31); */
/*     svst1_f32(pred16, &Ctrow(4, 16),  C41);     */
/*     svst1_f32(pred16, &Ctrow(5, 16),  C51);     */
/*     svst1_f32(pred16, &Ctrow(6, 16),  C61);     */
/*     svst1_f32(pred16, &Ctrow(7, 16),  C71); */
/*     svst1_f32(pred16, &Ctrow(8, 16),  C81); */
/*     svst1_f32(pred16, &Ctrow(9, 16),  C91); */
    
/*     if ( beta!=zero ) { */
/*       for ( j=0; j<nr; j++ ) */
/*         for ( i=0; i<mr; i++ ) { */
/*           Crow(i,j) = beta*Crow(i,j) + Ctrow(i,j); */
/* 	} */
/*     } */
    
/*     else { */
/*       for ( j=0; j<nr; j++ ) */
/*         for ( i=0; i<mr; i++ ) */
/*           Crow(i,j) = Ctrow(i,j); */
/*     } */
/*   } */
/*   else if ( (mr==MR) && (nr==NR) ) { */
/*     if (beta != zero) { */
/*       //Set Beta */
/*       Bta=svdup_n_f32(beta); */
      
/*       C0n = svld1_f32(pred16, &Crow(0,0)); */
/*       C1n = svld1_f32(pred16, &Crow(1,0)); */
/*       C2n = svld1_f32(pred16, &Crow(2,0)); */
/*       C3n = svld1_f32(pred16, &Crow(3,0)); */
/*       C4n = svld1_f32(pred16, &Crow(4,0)); */
/*       C5n = svld1_f32(pred16, &Crow(5,0)); */
/*       C6n = svld1_f32(pred16, &Crow(6,0)); */
/*       C7n = svld1_f32(pred16, &Crow(7,0)); */
/*       C8n = svld1_f32(pred16, &Crow(8,0)); */
/*       C9n = svld1_f32(pred16, &Crow(9,0)); */
      
/*       C00 = svmla_f32_z(pred16, C00, C0n, Bta); */
/*       C10 = svmla_f32_z(pred16, C10, C1n, Bta); */
/*       C20 = svmla_f32_z(pred16, C20, C2n, Bta); */
/*       C30 = svmla_f32_z(pred16, C30, C3n, Bta); */
/*       C40 = svmla_f32_z(pred16, C40, C4n, Bta); */
/*       C50 = svmla_f32_z(pred16, C50, C5n, Bta); */
/*       C60 = svmla_f32_z(pred16, C60, C6n, Bta); */
/*       C70 = svmla_f32_z(pred16, C70, C7n, Bta); */
/*       C80 = svmla_f32_z(pred16, C80, C8n, Bta); */
/*       C90 = svmla_f32_z(pred16, C90, C9n, Bta); */
            
/*       C0n = svld1_f32(pred16, &Crow(0,16)); */
/*       C1n = svld1_f32(pred16, &Crow(1,16)); */
/*       C2n = svld1_f32(pred16, &Crow(2,16)); */
/*       C3n = svld1_f32(pred16, &Crow(3,16)); */
/*       C4n = svld1_f32(pred16, &Crow(4,16)); */
/*       C5n = svld1_f32(pred16, &Crow(5,16)); */
/*       C6n = svld1_f32(pred16, &Crow(6,16)); */
/*       C7n = svld1_f32(pred16, &Crow(7,16)); */
/*       C8n = svld1_f32(pred16, &Crow(8,16)); */
/*       C9n = svld1_f32(pred16, &Crow(9,16)); */
	    
/*       C01 = svmla_f32_z(pred16, C01, C0n, Bta); */
/*       C11 = svmla_f32_z(pred16, C11, C1n, Bta); */
/*       C21 = svmla_f32_z(pred16, C21, C2n, Bta); */
/*       C31 = svmla_f32_z(pred16, C31, C3n, Bta); */
/*       C41 = svmla_f32_z(pred16, C41, C4n, Bta); */
/*       C51 = svmla_f32_z(pred16, C51, C5n, Bta); */
/*       C61 = svmla_f32_z(pred16, C61, C6n, Bta); */
/*       C71 = svmla_f32_z(pred16, C71, C7n, Bta); */
/*       C81 = svmla_f32_z(pred16, C81, C8n, Bta); */
/*       C91 = svmla_f32_z(pred16, C91, C9n, Bta); */
/*     } */
    
/*     svst1_f32(pred16, &Crow(0, 0), C00); */
/*     svst1_f32(pred16, &Crow(1, 0), C10); */
/*     svst1_f32(pred16, &Crow(2, 0), C20); */
/*     svst1_f32(pred16, &Crow(3, 0), C30); */
/*     svst1_f32(pred16, &Crow(4, 0), C40); */
/*     svst1_f32(pred16, &Crow(5, 0), C50); */
/*     svst1_f32(pred16, &Crow(6, 0), C60); */
/*     svst1_f32(pred16, &Crow(7, 0), C70); */
/*     svst1_f32(pred16, &Crow(8, 0), C80); */
/*     svst1_f32(pred16, &Crow(9, 0), C90); */

/*     svst1_f32(pred16, &Crow(0, 16), C01); */
/*     svst1_f32(pred16, &Crow(1, 16), C11); */
/*     svst1_f32(pred16, &Crow(2, 16), C21); */
/*     svst1_f32(pred16, &Crow(3, 16), C31); */
/*     svst1_f32(pred16, &Crow(4, 16), C41); */
/*     svst1_f32(pred16, &Crow(5, 16), C51); */
/*     svst1_f32(pred16, &Crow(6, 16), C61); */
/*     svst1_f32(pred16, &Crow(7, 16), C71); */
/*     svst1_f32(pred16, &Crow(8, 16), C81); */
/*     svst1_f32(pred16, &Crow(9, 16), C91); */

/*   } */
/*   else { */
/*     printf("Error: Incorrect use of 4x16 micro-kernel with %d x %d block\n", mr, nr); */
/*     exit(-1); */
/*   } */
/* } */

void gemm_microkernel_neon_sve_fixed_10x32_fp32( int mr, int nr, int kc, float alpha, float *Ar, float *Br, float beta, float *C, int ldC ) {
/*
  BLIS GEMM microkernel, computes the product Cr := Cr + Ar * Br
  Specific: only for MRxNR = 4x16
*/
  //SET_MR_NR(10, 32);
  
  int         i, j, k, baseB = 0, baseA = 0, ldCt = NR, Amr, Bnr;

  //Registers for C = (20)
  svfloat32_t C00, C10, C20, C30, C40, C50, C60, C70, C80, C90,
              C01, C11, C21, C31, C41, C51, C61, C71, C81, C91;

  //Other auxiliar Registers (11)
  svfloat32_t  B0,  B1,  A0, A1, Bta, C0n, C1n, C2n, C3n, C4n, C5n, C6n, C7n, C8n, C9n;  

  #define C7n A0
  #define C8n B0
  #define C9n B1

  float   zero = 0.0, one = 1.0, *Aptr, *Bptr, Ctmp[MR * NR];
  svbool_t pred16 = svwhilelt_b32_u32(0, 16);
  
  if ( kc==0 )
    return;
    
  //Set to 0
  C00=svdup_n_f32(0); C01=svdup_n_f32(0); 
  C10=svdup_n_f32(0); C11=svdup_n_f32(0); 
  C20=svdup_n_f32(0); C21=svdup_n_f32(0); 
  C30=svdup_n_f32(0); C31=svdup_n_f32(0); 
  C40=svdup_n_f32(0); C41=svdup_n_f32(0); 
  C50=svdup_n_f32(0); C51=svdup_n_f32(0); 
  C60=svdup_n_f32(0); C61=svdup_n_f32(0); 
  C70=svdup_n_f32(0); C71=svdup_n_f32(0);
  C80=svdup_n_f32(0); C81=svdup_n_f32(0);
  C90=svdup_n_f32(0); C91=svdup_n_f32(0); 
  
  Aptr = &Ar[0];
  Amr  = MR;
  Bptr = &Br[0];
  Bnr  = NR;

  if ( alpha!=zero ) {
    for ( k=0; k<kc; k++ ) {
      // Load colums/rows of A/B for current iteration
      B0 = svld1_f32(pred16, &Bptr[baseB]);
      B1 = svld1_f32(pred16, &Bptr[baseB + 16]);
      
      A0=svdup_n_f32(Aptr[baseA + 0]);
      C00 = svmla_f32_z(pred16, C00, B0, A0);
      C01 = svmla_f32_z(pred16, C01, B1, A0);
      
      A1=svdup_n_f32(Aptr[baseA + 1]);
      C10 = svmla_f32_z(pred16, C10, B0, A1);
      C11 = svmla_f32_z(pred16, C11, B1, A1);
      
      A0=svdup_n_f32(Aptr[baseA + 2]);
      C20 = svmla_f32_z(pred16, C20, B0, A0);
      C21 = svmla_f32_z(pred16, C21, B1, A0);
      
      A1=svdup_n_f32(Aptr[baseA + 3]);
      C30 = svmla_f32_z(pred16, C30, B0, A1);
      C31 = svmla_f32_z(pred16, C31, B1, A1);
      
      A0=svdup_n_f32(Aptr[baseA + 4]);
      C40 = svmla_f32_z(pred16, C40, B0, A0);
      C41 = svmla_f32_z(pred16, C41, B1, A0);
      
      A1=svdup_n_f32(Aptr[baseA + 5]);
      C50 = svmla_f32_z(pred16, C50, B0, A1);
      C51 = svmla_f32_z(pred16, C51, B1, A1);
      
      A0=svdup_n_f32(Aptr[baseA + 6]);
      C60 = svmla_f32_z(pred16, C60, B0, A0);
      C61 = svmla_f32_z(pred16, C61, B1, A0);
      
      A1=svdup_n_f32(Aptr[baseA + 7]);
      C70 = svmla_f32_z(pred16, C70, B0, A1);
      C71 = svmla_f32_z(pred16, C71, B1, A1);
      
      A0=svdup_n_f32(Aptr[baseA + 8]);
      C80 = svmla_f32_z(pred16, C80, B0, A0);
      C81 = svmla_f32_z(pred16, C81, B1, A0);
      
      A1=svdup_n_f32(Aptr[baseA + 9]);
      C90 = svmla_f32_z(pred16, C90, B0, A1);
      C91 = svmla_f32_z(pred16, C91, B1, A1);
      
      baseA = baseA + Amr;
      baseB = baseB + Bnr;
    }
    
    if ( alpha==-one ) {
      C00 = svmul_n_f32_x(pred16, C00, -1.0);
      C10 = svmul_n_f32_x(pred16, C10, -1.0);
      C20 = svmul_n_f32_x(pred16, C20, -1.0);
      C30 = svmul_n_f32_x(pred16, C30, -1.0);
      C40 = svmul_n_f32_x(pred16, C40, -1.0);
      C50 = svmul_n_f32_x(pred16, C50, -1.0);
      C60 = svmul_n_f32_x(pred16, C60, -1.0);
      C70 = svmul_n_f32_x(pred16, C70, -1.0);
      C80 = svmul_n_f32_x(pred16, C80, -1.0);
      C90 = svmul_n_f32_x(pred16, C90, -1.0);
      
      C01 = svmul_n_f32_x(pred16, C01, -1.0);
      C11 = svmul_n_f32_x(pred16, C11, -1.0);
      C21 = svmul_n_f32_x(pred16, C21, -1.0);
      C31 = svmul_n_f32_x(pred16, C31, -1.0);
      C41 = svmul_n_f32_x(pred16, C40, -1.0);
      C51 = svmul_n_f32_x(pred16, C51, -1.0);
      C61 = svmul_n_f32_x(pred16, C61, -1.0);
      C71 = svmul_n_f32_x(pred16, C71, -1.0);
      C81 = svmul_n_f32_x(pred16, C81, -1.0);
      C91 = svmul_n_f32_x(pred16, C91, -1.0);
    } else if ( alpha!=one ) {
      C00 = svmul_n_f32_x(pred16, C00, alpha);
      C10 = svmul_n_f32_x(pred16, C10, alpha);
      C20 = svmul_n_f32_x(pred16, C20, alpha);
      C30 = svmul_n_f32_x(pred16, C30, alpha);
      C40 = svmul_n_f32_x(pred16, C40, alpha);
      C50 = svmul_n_f32_x(pred16, C50, alpha);
      C60 = svmul_n_f32_x(pred16, C60, alpha);
      C70 = svmul_n_f32_x(pred16, C70, alpha);
      C80 = svmul_n_f32_x(pred16, C80, alpha);
      C90 = svmul_n_f32_x(pred16, C90, alpha);
      
      C01 = svmul_n_f32_x(pred16, C01, alpha);
      C11 = svmul_n_f32_x(pred16, C11, alpha);
      C21 = svmul_n_f32_x(pred16, C21, alpha);
      C31 = svmul_n_f32_x(pred16, C31, alpha);
      C41 = svmul_n_f32_x(pred16, C41, alpha);
      C51 = svmul_n_f32_x(pred16, C51, alpha);
      C61 = svmul_n_f32_x(pred16, C61, alpha);
      C71 = svmul_n_f32_x(pred16, C71, alpha);
      C81 = svmul_n_f32_x(pred16, C81, alpha);
      C91 = svmul_n_f32_x(pred16, C91, alpha);
    }
  }
  
  if ( (mr<MR) || (nr<NR) ) {
    svst1_f32(pred16, &Ctrow(0, 0),  C00);
    svst1_f32(pred16, &Ctrow(1, 0),  C10);
    svst1_f32(pred16, &Ctrow(2, 0),  C20);
    svst1_f32(pred16, &Ctrow(3, 0),  C30);
    svst1_f32(pred16, &Ctrow(4, 0),  C40);
    svst1_f32(pred16, &Ctrow(5, 0),  C50);
    svst1_f32(pred16, &Ctrow(6, 0),  C60);
    svst1_f32(pred16, &Ctrow(7, 0),  C70);
    svst1_f32(pred16, &Ctrow(8, 0),  C80);
    svst1_f32(pred16, &Ctrow(9, 0),  C90);
    
    svst1_f32(pred16, &Ctrow(0, 16),  C01);
    svst1_f32(pred16, &Ctrow(1, 16),  C11);
    svst1_f32(pred16, &Ctrow(2, 16),  C21);
    svst1_f32(pred16, &Ctrow(3, 16),  C31);
    svst1_f32(pred16, &Ctrow(4, 16),  C41);
    svst1_f32(pred16, &Ctrow(5, 16),  C51);
    svst1_f32(pred16, &Ctrow(6, 16),  C61);
    svst1_f32(pred16, &Ctrow(7, 16),  C71);
    svst1_f32(pred16, &Ctrow(8, 16),  C81);
    svst1_f32(pred16, &Ctrow(9, 16),  C91);
    
    if ( beta!=zero ) {
      for ( j=0; j<nr; j++ )
        for ( i=0; i<mr; i++ ) {
          Crow(i,j) = beta*Crow(i,j) + Ctrow(i,j);
  	}
    }
    
    else {
      for ( j=0; j<nr; j++ )
        for ( i=0; i<mr; i++ )
          Crow(i,j) = Ctrow(i,j);
    }
  }
  else if ( (mr==MR) && (nr==NR) ) {
    if (beta != zero) {
      //Set Beta
      Bta=svdup_n_f32(beta);
      
      C0n = svld1_f32(pred16, &Crow(0,0));
      C1n = svld1_f32(pred16, &Crow(1,0));
      C2n = svld1_f32(pred16, &Crow(2,0));
      C3n = svld1_f32(pred16, &Crow(3,0));
      C4n = svld1_f32(pred16, &Crow(4,0));
      C5n = svld1_f32(pred16, &Crow(5,0));
      C6n = svld1_f32(pred16, &Crow(6,0));
      C7n = svld1_f32(pred16, &Crow(7,0));
      C8n = svld1_f32(pred16, &Crow(8,0));
      C9n = svld1_f32(pred16, &Crow(9,0));
      
      C00 = svmla_f32_z(pred16, C00, C0n, Bta);
      C10 = svmla_f32_z(pred16, C10, C1n, Bta);
      C20 = svmla_f32_z(pred16, C20, C2n, Bta);
      C30 = svmla_f32_z(pred16, C30, C3n, Bta);
      C40 = svmla_f32_z(pred16, C40, C4n, Bta);
      C50 = svmla_f32_z(pred16, C50, C5n, Bta);
      C60 = svmla_f32_z(pred16, C60, C6n, Bta);
      C70 = svmla_f32_z(pred16, C70, C7n, Bta);
      C80 = svmla_f32_z(pred16, C80, C8n, Bta);
      C90 = svmla_f32_z(pred16, C90, C9n, Bta);
            
      C0n = svld1_f32(pred16, &Crow(0,16));
      C1n = svld1_f32(pred16, &Crow(1,16));
      C2n = svld1_f32(pred16, &Crow(2,16));
      C3n = svld1_f32(pred16, &Crow(3,16));
      C4n = svld1_f32(pred16, &Crow(4,16));
      C5n = svld1_f32(pred16, &Crow(5,16));
      C6n = svld1_f32(pred16, &Crow(6,16));
      C7n = svld1_f32(pred16, &Crow(7,16));
      C8n = svld1_f32(pred16, &Crow(8,16));
      C9n = svld1_f32(pred16, &Crow(9,16));
	    
      C01 = svmla_f32_z(pred16, C01, C0n, Bta);
      C11 = svmla_f32_z(pred16, C11, C1n, Bta);
      C21 = svmla_f32_z(pred16, C21, C2n, Bta);
      C31 = svmla_f32_z(pred16, C31, C3n, Bta);
      C41 = svmla_f32_z(pred16, C41, C4n, Bta);
      C51 = svmla_f32_z(pred16, C51, C5n, Bta);
      C61 = svmla_f32_z(pred16, C61, C6n, Bta);
      C71 = svmla_f32_z(pred16, C71, C7n, Bta);
      C81 = svmla_f32_z(pred16, C81, C8n, Bta);
      C91 = svmla_f32_z(pred16, C91, C9n, Bta);
    }
    
    svst1_f32(pred16, &Crow(0, 0), C00);
    svst1_f32(pred16, &Crow(1, 0), C10);
    svst1_f32(pred16, &Crow(2, 0), C20);
    svst1_f32(pred16, &Crow(3, 0), C30);
    svst1_f32(pred16, &Crow(4, 0), C40);
    svst1_f32(pred16, &Crow(5, 0), C50);
    svst1_f32(pred16, &Crow(6, 0), C60);
    svst1_f32(pred16, &Crow(7, 0), C70);
    svst1_f32(pred16, &Crow(8, 0), C80);
    svst1_f32(pred16, &Crow(9, 0), C90);

    svst1_f32(pred16, &Crow(0, 16), C01);
    svst1_f32(pred16, &Crow(1, 16), C11);
    svst1_f32(pred16, &Crow(2, 16), C21);
    svst1_f32(pred16, &Crow(3, 16), C31);
    svst1_f32(pred16, &Crow(4, 16), C41);
    svst1_f32(pred16, &Crow(5, 16), C51);
    svst1_f32(pred16, &Crow(6, 16), C61);
    svst1_f32(pred16, &Crow(7, 16), C71);
    svst1_f32(pred16, &Crow(8, 16), C81);
    svst1_f32(pred16, &Crow(9, 16), C91);

  }
  else {
    printf("Error: Incorrect use of 4x16 micro-kernel with %d x %d block\n", mr, nr);
    exit(-1);
  }
}

void gemm_microkernel_neon_sve_fixed_12x32_fp32( int mr, int nr, int kc, float alpha, float *Ar, float *Br, float beta, float *C, int ldC ) {
/*
  BLIS GEMM microkernel, computes the product Cr := Cr + Ar * Br
  Specific: only for MRxNR = 4x16
*/
  //SET_MR_NR(12, 32);
  
  int         i, j, k, baseB = 0, baseA = 0, ldCt = NR, Amr, Bnr;

  //Registers for C = (24)
  svfloat32_t C00, C10, C20, C30, C40, C50, C60, C70, C80, C90, C100, C110,
              C01, C11, C21, C31, C41, C51, C61, C71, C81, C91, C101, C111;

  //Other auxiliar Registers (11)
  svfloat32_t  B0,  B1,  A0, A1, C0n, C1n, Bta; /*Bta; /*,C0n, C1n, C2n, C3n, C4n, C5n, C6n, C10n, C11n;  */

  float   zero = 0.0, one = 1.0, *Aptr, *Bptr, Ctmp[MR * NR];

  svbool_t pred16 = svwhilelt_b32_u32(0, 16);
  
  if ( kc==0 )
    return;


  //Set to 0
  C00=svdup_n_f32(0); C01=svdup_n_f32(0); 
  C10=svdup_n_f32(0); C11=svdup_n_f32(0); 
  C20=svdup_n_f32(0); C21=svdup_n_f32(0); 
  C30=svdup_n_f32(0); C31=svdup_n_f32(0); 
  C40=svdup_n_f32(0); C41=svdup_n_f32(0); 
  C50=svdup_n_f32(0); C51=svdup_n_f32(0); 
  C60=svdup_n_f32(0); C61=svdup_n_f32(0); 
  C70=svdup_n_f32(0); C71=svdup_n_f32(0);
  C80=svdup_n_f32(0); C81=svdup_n_f32(0);
  C90=svdup_n_f32(0); C91=svdup_n_f32(0);
  C100=svdup_n_f32(0); C101=svdup_n_f32(0); 
  C110=svdup_n_f32(0); C111=svdup_n_f32(0);
  
  Aptr = &Ar[0];
  Amr  = MR;
  Bptr = &Br[0];
  Bnr  = NR;
  
  if ( alpha!=zero ) {
    for ( k=0; k<kc; k++ ) {
      // Load colums/rows of A/B for current iteration
      B0 = svld1_f32(pred16, &Bptr[baseB]);
      B1 = svld1_f32(pred16, &Bptr[baseB + 16]);
      
      A0=svdup_n_f32(Aptr[baseA + 0]);
      C00 = svmla_f32_z(pred16, C00, B0, A0);
      C01 = svmla_f32_z(pred16, C01, B1, A0);
      
      A1=svdup_n_f32(Aptr[baseA + 1]);
      C10 = svmla_f32_z(pred16, C10, B0, A1);
      C11 = svmla_f32_z(pred16, C11, B1, A1);
      
      A0=svdup_n_f32(Aptr[baseA + 2]);
      C20 = svmla_f32_z(pred16, C20, B0, A0);
      C21 = svmla_f32_z(pred16, C21, B1, A0);
      
      A1=svdup_n_f32(Aptr[baseA + 3]);
      C30 = svmla_f32_z(pred16, C30, B0, A1);
      C31 = svmla_f32_z(pred16, C31, B1, A1);
      
      A0=svdup_n_f32(Aptr[baseA + 4]);
      C40 = svmla_f32_z(pred16, C40, B0, A0);
      C41 = svmla_f32_z(pred16, C41, B1, A0);
      
      A1=svdup_n_f32(Aptr[baseA + 5]);
      C50 = svmla_f32_z(pred16, C50, B0, A1);
      C51 = svmla_f32_z(pred16, C51, B1, A1);
      
      A0=svdup_n_f32(Aptr[baseA + 6]);
      C60 = svmla_f32_z(pred16, C60, B0, A0);
      C61 = svmla_f32_z(pred16, C61, B1, A0);
      
      A1=svdup_n_f32(Aptr[baseA + 7]);
      C70 = svmla_f32_z(pred16, C70, B0, A1);
      C71 = svmla_f32_z(pred16, C71, B1, A1);
      
      A0=svdup_n_f32(Aptr[baseA + 8]);
      C80 = svmla_f32_z(pred16, C80, B0, A0);
      C81 = svmla_f32_z(pred16, C81, B1, A0);
      
      A1=svdup_n_f32(Aptr[baseA + 9]);
      C90 = svmla_f32_z(pred16, C90, B0, A1);
      C91 = svmla_f32_z(pred16, C91, B1, A1);
      
      A0=svdup_n_f32(Aptr[baseA + 10]);
      C100 = svmla_f32_z(pred16, C100, B0, A0);
      C101 = svmla_f32_z(pred16, C101, B1, A0);

      A1=svdup_n_f32(Aptr[baseA + 11]);
      C110 = svmla_f32_z(pred16, C110, B0, A1);
      C111 = svmla_f32_z(pred16, C111, B1, A1);

      baseA = baseA + Amr;
      baseB = baseB + Bnr;
    }
    
    if ( alpha==-one ) {
      C00 = svmul_n_f32_x(pred16, C00, -1.0);
      C10 = svmul_n_f32_x(pred16, C10, -1.0);
      C20 = svmul_n_f32_x(pred16, C20, -1.0);
      C30 = svmul_n_f32_x(pred16, C30, -1.0);
      C40 = svmul_n_f32_x(pred16, C40, -1.0);
      C50 = svmul_n_f32_x(pred16, C50, -1.0);
      C60 = svmul_n_f32_x(pred16, C60, -1.0);
      C70 = svmul_n_f32_x(pred16, C70, -1.0);
      C80 = svmul_n_f32_x(pred16, C80, -1.0);
      C90 = svmul_n_f32_x(pred16, C90, -1.0);
      C100 = svmul_n_f32_x(pred16, C100, -1.0);
      C110 = svmul_n_f32_x(pred16, C110, -1.0);
      
      C01 = svmul_n_f32_x(pred16, C01, -1.0);
      C11 = svmul_n_f32_x(pred16, C11, -1.0);
      C21 = svmul_n_f32_x(pred16, C21, -1.0);
      C31 = svmul_n_f32_x(pred16, C31, -1.0);
      C41 = svmul_n_f32_x(pred16, C40, -1.0);
      C51 = svmul_n_f32_x(pred16, C51, -1.0);
      C61 = svmul_n_f32_x(pred16, C61, -1.0);
      C71 = svmul_n_f32_x(pred16, C71, -1.0);
      C81 = svmul_n_f32_x(pred16, C81, -1.0);
      C91 = svmul_n_f32_x(pred16, C91, -1.0);
      C101 = svmul_n_f32_x(pred16, C101, -1.0);
      C111 = svmul_n_f32_x(pred16, C111, -1.0);
    } else if ( alpha!=one ) {
      C00 = svmul_n_f32_x(pred16, C00, alpha);
      C10 = svmul_n_f32_x(pred16, C10, alpha);
      C20 = svmul_n_f32_x(pred16, C20, alpha);
      C30 = svmul_n_f32_x(pred16, C30, alpha);
      C40 = svmul_n_f32_x(pred16, C40, alpha);
      C50 = svmul_n_f32_x(pred16, C50, alpha);
      C60 = svmul_n_f32_x(pred16, C60, alpha);
      C70 = svmul_n_f32_x(pred16, C70, alpha);
      C80 = svmul_n_f32_x(pred16, C80, alpha);
      C90 = svmul_n_f32_x(pred16, C90, alpha);
      C100 = svmul_n_f32_x(pred16, C100, alpha);
      C110 = svmul_n_f32_x(pred16, C110, alpha);
      
      C01 = svmul_n_f32_x(pred16, C01, alpha);
      C11 = svmul_n_f32_x(pred16, C11, alpha);
      C21 = svmul_n_f32_x(pred16, C21, alpha);
      C31 = svmul_n_f32_x(pred16, C31, alpha);
      C41 = svmul_n_f32_x(pred16, C41, alpha);
      C51 = svmul_n_f32_x(pred16, C51, alpha);
      C61 = svmul_n_f32_x(pred16, C61, alpha);
      C71 = svmul_n_f32_x(pred16, C71, alpha);
      C81 = svmul_n_f32_x(pred16, C81, alpha);
      C91 = svmul_n_f32_x(pred16, C91, alpha);
      C101 = svmul_n_f32_x(pred16, C101, alpha);
      C111 = svmul_n_f32_x(pred16, C111, alpha);
    }
  }
  
  if ( (mr<MR) || (nr<NR) ) {
    svst1_f32(pred16, &Ctrow(0, 0),  C00);
    svst1_f32(pred16, &Ctrow(1, 0),  C10);
    svst1_f32(pred16, &Ctrow(2, 0),  C20);
    svst1_f32(pred16, &Ctrow(3, 0),  C30);
    svst1_f32(pred16, &Ctrow(4, 0),  C40);
    svst1_f32(pred16, &Ctrow(5, 0),  C50);
    svst1_f32(pred16, &Ctrow(6, 0),  C60);
    svst1_f32(pred16, &Ctrow(7, 0),  C70);
    svst1_f32(pred16, &Ctrow(8, 0),  C80);
    svst1_f32(pred16, &Ctrow(9, 0),  C90);
    svst1_f32(pred16, &Ctrow(10, 0),  C100);
    svst1_f32(pred16, &Ctrow(11, 0),  C110);
    
    svst1_f32(pred16, &Ctrow(0, 16),  C01);
    svst1_f32(pred16, &Ctrow(1, 16),  C11);
    svst1_f32(pred16, &Ctrow(2, 16),  C21);
    svst1_f32(pred16, &Ctrow(3, 16),  C31);
    svst1_f32(pred16, &Ctrow(4, 16),  C41);
    svst1_f32(pred16, &Ctrow(5, 16),  C51);
    svst1_f32(pred16, &Ctrow(6, 16),  C61);
    svst1_f32(pred16, &Ctrow(7, 16),  C71);
    svst1_f32(pred16, &Ctrow(8, 16),  C81);
    svst1_f32(pred16, &Ctrow(9, 16),  C91);
    svst1_f32(pred16, &Ctrow(10, 16),  C101);
    svst1_f32(pred16, &Ctrow(11, 16),  C111);
    
    if ( beta!=zero ) {
      for ( j=0; j<nr; j++ )
        for ( i=0; i<mr; i++ ) {
          Crow(i,j) = beta*Crow(i,j) + Ctrow(i,j);
  	}
    }
    
    else {
      for ( j=0; j<nr; j++ )
        for ( i=0; i<mr; i++ )
          Crow(i,j) = Ctrow(i,j);
    }
  }
  else if ( (mr==MR) && (nr==NR) ) {
    if (beta != zero) {
      //Set Beta
      Bta=svdup_n_f32(beta);

      C0n = svld1_f32(pred16, &Crow(0,0));
      C00 = svmla_f32_z(pred16, C00, C0n, Bta);

      C1n = svld1_f32(pred16, &Crow(1,0));
      C10 = svmla_f32_z(pred16, C10, C1n, Bta);

      C0n = svld1_f32(pred16, &Crow(2,0));
      C20 = svmla_f32_z(pred16, C20, C0n, Bta);

      C1n = svld1_f32(pred16, &Crow(3,0));
      C30 = svmla_f32_z(pred16, C30, C1n, Bta);

      C0n = svld1_f32(pred16, &Crow(4,0));
      C40 = svmla_f32_z(pred16, C40, C0n, Bta);

      C1n = svld1_f32(pred16, &Crow(5,0));
      C50 = svmla_f32_z(pred16, C50, C1n, Bta);

      C0n = svld1_f32(pred16, &Crow(6,0));
      C60 = svmla_f32_z(pred16, C60, C0n, Bta);

      C1n = svld1_f32(pred16, &Crow(7,0));
      C70 = svmla_f32_z(pred16, C70, C1n, Bta);

      C0n = svld1_f32(pred16, &Crow(8,0));
      C80 = svmla_f32_z(pred16, C80, C0n, Bta);

      C1n = svld1_f32(pred16, &Crow(9,0));
      C90 = svmla_f32_z(pred16, C90, C1n, Bta);

      C0n = svld1_f32(pred16, &Crow(10,0));
      C100 = svmla_f32_z(pred16, C100, C0n, Bta);

      C1n = svld1_f32(pred16, &Crow(11,0));
      C110 = svmla_f32_z(pred16, C110, C1n, Bta);


      C0n = svld1_f32(pred16, &Crow(0, 16));
      C01 = svmla_f32_z(pred16, C01, C0n, Bta);

      C1n = svld1_f32(pred16, &Crow(1, 16));
      C11 = svmla_f32_z(pred16, C11, C1n, Bta);

      C0n = svld1_f32(pred16, &Crow(2, 16));
      C21 = svmla_f32_z(pred16, C21, C0n, Bta);

      C1n = svld1_f32(pred16, &Crow(3, 16));
      C31 = svmla_f32_z(pred16, C31, C1n, Bta);

      C0n = svld1_f32(pred16, &Crow(4, 16));
      C41 = svmla_f32_z(pred16, C41, C0n, Bta);

      C1n = svld1_f32(pred16, &Crow(5, 16));
      C51 = svmla_f32_z(pred16, C51, C1n, Bta);

      C0n = svld1_f32(pred16, &Crow(6, 16));
      C61 = svmla_f32_z(pred16, C61, C0n, Bta);

      C1n = svld1_f32(pred16, &Crow(7, 16));
      C71 = svmla_f32_z(pred16, C71, C1n, Bta);

      C0n = svld1_f32(pred16, &Crow(8, 16));
      C81 = svmla_f32_z(pred16, C81, C0n, Bta);

      C1n = svld1_f32(pred16, &Crow(9, 16));
      C91 = svmla_f32_z(pred16, C91, C1n, Bta);

      C0n = svld1_f32(pred16, &Crow(10, 16));
      C101 = svmla_f32_z(pred16, C101, C0n, Bta);

      C1n = svld1_f32(pred16, &Crow(11, 16));
      C111 = svmla_f32_z(pred16, C111, C1n, Bta);

    }
    /* if (beta != zero) { */
    /*   //Set Beta */
    /*   Bta=svdup_n_f32(beta); */
      
    /*   C0n = svld1_f32(pred16, &Crow(0,0)); */
    /*   C1n = svld1_f32(pred16, &Crow(1,0)); */
    /*   C2n = svld1_f32(pred16, &Crow(2,0)); */
    /*   C3n = svld1_f32(pred16, &Crow(3,0)); */
    /*   C4n = svld1_f32(pred16, &Crow(4,0)); */
    /*   C5n = svld1_f32(pred16, &Crow(5,0)); */
    /*   C6n = svld1_f32(pred16, &Crow(6,0)); */
    /*   C7n = svld1_f32(pred16, &Crow(7,0)); */
    /*   C8n = svld1_f32(pred16, &Crow(8,0)); */
    /*   C9n = svld1_f32(pred16, &Crow(9,0)); */
    /*   C10n = svld1_f32(pred16, &Crow(10,0)); */
    /*   C11n = svld1_f32(pred16, &Crow(11,0)); */
      
    /*   C00 = svmla_f32_z(pred16, C00, C0n, Bta); */
    /*   C10 = svmla_f32_z(pred16, C10, C1n, Bta); */
    /*   C20 = svmla_f32_z(pred16, C20, C2n, Bta); */
    /*   C30 = svmla_f32_z(pred16, C30, C3n, Bta); */
    /*   C40 = svmla_f32_z(pred16, C40, C4n, Bta); */
    /*   C50 = svmla_f32_z(pred16, C50, C5n, Bta); */
    /*   C60 = svmla_f32_z(pred16, C60, C6n, Bta); */
    /*   C70 = svmla_f32_z(pred16, C70, C7n, Bta); */
    /*   C80 = svmla_f32_z(pred16, C80, C8n, Bta); */
    /*   C90 = svmla_f32_z(pred16, C90, C9n, Bta); */
    /*   C100 = svmla_f32_z(pred16, C100, C10n, Bta); */
    /*   C110 = svmla_f32_z(pred16, C110, C11n, Bta); */
            
    /*   C0n = svld1_f32(pred16, &Crow(0,16)); */
    /*   C1n = svld1_f32(pred16, &Crow(1,16)); */
    /*   C2n = svld1_f32(pred16, &Crow(2,16)); */
    /*   C3n = svld1_f32(pred16, &Crow(3,16)); */
    /*   C4n = svld1_f32(pred16, &Crow(4,16)); */
    /*   C5n = svld1_f32(pred16, &Crow(5,16)); */
    /*   C6n = svld1_f32(pred16, &Crow(6,16)); */
    /*   C7n = svld1_f32(pred16, &Crow(7,16)); */
    /*   C8n = svld1_f32(pred16, &Crow(8,16)); */
    /*   C9n = svld1_f32(pred16, &Crow(9,16)); */
    /*   C10n = svld1_f32(pred16, &Crow(10,16)); */
    /*   C11n = svld1_f32(pred16, &Crow(11,16)); */
	    
    /*   C01 = svmla_f32_z(pred16, C01, C0n, Bta); */
    /*   C11 = svmla_f32_z(pred16, C11, C1n, Bta); */
    /*   C21 = svmla_f32_z(pred16, C21, C2n, Bta); */
    /*   C31 = svmla_f32_z(pred16, C31, C3n, Bta); */
    /*   C41 = svmla_f32_z(pred16, C41, C4n, Bta); */
    /*   C51 = svmla_f32_z(pred16, C51, C5n, Bta); */
    /*   C61 = svmla_f32_z(pred16, C61, C6n, Bta); */
    /*   C71 = svmla_f32_z(pred16, C71, C7n, Bta); */
    /*   C81 = svmla_f32_z(pred16, C81, C8n, Bta); */
    /*   C91 = svmla_f32_z(pred16, C91, C9n, Bta); */
    /*   C101 = svmla_f32_z(pred16, C101, C10n, Bta); */
    /*   C111 = svmla_f32_z(pred16, C111, C11n, Bta); */
    /* } */
    
    svst1_f32(pred16, &Crow(0, 0), C00);
    svst1_f32(pred16, &Crow(1, 0), C10);
    svst1_f32(pred16, &Crow(2, 0), C20);
    svst1_f32(pred16, &Crow(3, 0), C30);
    svst1_f32(pred16, &Crow(4, 0), C40);
    svst1_f32(pred16, &Crow(5, 0), C50);
    svst1_f32(pred16, &Crow(6, 0), C60);
    svst1_f32(pred16, &Crow(7, 0), C70);
    svst1_f32(pred16, &Crow(8, 0), C80);
    svst1_f32(pred16, &Crow(9, 0), C90);
    svst1_f32(pred16, &Crow(10, 0), C100);
    svst1_f32(pred16, &Crow(11, 0), C110);

    svst1_f32(pred16, &Crow(0, 16), C01);
    svst1_f32(pred16, &Crow(1, 16), C11);
    svst1_f32(pred16, &Crow(2, 16), C21);
    svst1_f32(pred16, &Crow(3, 16), C31);
    svst1_f32(pred16, &Crow(4, 16), C41);
    svst1_f32(pred16, &Crow(5, 16), C51);
    svst1_f32(pred16, &Crow(6, 16), C61);
    svst1_f32(pred16, &Crow(7, 16), C71);
    svst1_f32(pred16, &Crow(8, 16), C81);
    svst1_f32(pred16, &Crow(9, 16), C91);
    svst1_f32(pred16, &Crow(10, 16), C101);
    svst1_f32(pred16, &Crow(11, 16), C111);

  }
  else {
    printf("Error: Incorrect use of 4x16 micro-kernel with %d x %d block\n", mr, nr);
    exit(-1);
  }
}


void gemm_microkernel_neon_sve_fixed_14x32_fp32( int mr, int nr, int kc, float alpha, float *Ar, float *Br, float beta, float *C, int ldC ) {
/*
  BLIS GEMM microkernel, computes the product Cr := Cr + Ar * Br
  Specific: only for MRxNR = 4x16
*/
  //SET_MR_NR(14, 32);
  
  int         i, j, k, baseB = 0, baseA = 0, ldCt = NR, Amr, Bnr;

  //Registers for C = (28)
  svfloat32_t C00, C10, C20, C30, C40, C50, C60, C70, C80, C90, C100, C110, C120, C130,
              C01, C11, C21, C31, C41, C51, C61, C71, C81, C91, C101, C111, C121, C131;

  //Other auxiliar Registers (11)
  svfloat32_t  B0,  B1,  A0, A1; //Bta, C0n, C1n;  

  svfloat32_t C3p, C4p, C5p, C6p, C7p, C8p, C9p, C10p, C11p, C12p, C13p;
  
  #define C0p B0
  #define C1p B1
  #define C2p A0
  #define Bta A1
  
  float   zero = 0.0, one = 1.0, *Aptr, *Bptr, Ctmp[MR * NR];
  svbool_t pred16 = svwhilelt_b32_u32(0, 16);
  
  if ( kc==0 )
    return;
    
  //Set to 0
  C00=svdup_n_f32(0); C01=svdup_n_f32(0); 
  C10=svdup_n_f32(0); C11=svdup_n_f32(0); 
  C20=svdup_n_f32(0); C21=svdup_n_f32(0); 
  C30=svdup_n_f32(0); C31=svdup_n_f32(0); 
  C40=svdup_n_f32(0); C41=svdup_n_f32(0); 
  C50=svdup_n_f32(0); C51=svdup_n_f32(0); 
  C60=svdup_n_f32(0); C61=svdup_n_f32(0); 
  C70=svdup_n_f32(0); C71=svdup_n_f32(0);
  C80=svdup_n_f32(0); C81=svdup_n_f32(0);
  C90=svdup_n_f32(0); C91=svdup_n_f32(0);
  C100=svdup_n_f32(0); C101=svdup_n_f32(0); 
  C110=svdup_n_f32(0); C111=svdup_n_f32(0);
  C120=svdup_n_f32(0); C121=svdup_n_f32(0); 
  C130=svdup_n_f32(0); C131=svdup_n_f32(0);
  
  Aptr = &Ar[0];
  Amr  = MR;
  Bptr = &Br[0];
  Bnr  = NR;
  
  if ( alpha!=zero ) {
    for ( k=0; k<kc; k++ ) {
      // Load colums/rows of A/B for current iteration
      B0 = svld1_f32(pred16, &Bptr[baseB]);
      B1 = svld1_f32(pred16, &Bptr[baseB + 16]);
      
      A0=svdup_n_f32(Aptr[baseA + 0]);
      C00 = svmla_f32_z(pred16, C00, B0, A0);
      C01 = svmla_f32_z(pred16, C01, B1, A0);
      
      A1=svdup_n_f32(Aptr[baseA + 1]);
      C10 = svmla_f32_z(pred16, C10, B0, A1);
      C11 = svmla_f32_z(pred16, C11, B1, A1);
      
      A0=svdup_n_f32(Aptr[baseA + 2]);
      C20 = svmla_f32_z(pred16, C20, B0, A0);
      C21 = svmla_f32_z(pred16, C21, B1, A0);
      
      A1=svdup_n_f32(Aptr[baseA + 3]);
      C30 = svmla_f32_z(pred16, C30, B0, A1);
      C31 = svmla_f32_z(pred16, C31, B1, A1);
      
      A0=svdup_n_f32(Aptr[baseA + 4]);
      C40 = svmla_f32_z(pred16, C40, B0, A0);
      C41 = svmla_f32_z(pred16, C41, B1, A0);
      
      A1=svdup_n_f32(Aptr[baseA + 5]);
      C50 = svmla_f32_z(pred16, C50, B0, A1);
      C51 = svmla_f32_z(pred16, C51, B1, A1);
      
      A0=svdup_n_f32(Aptr[baseA + 6]);
      C60 = svmla_f32_z(pred16, C60, B0, A0);
      C61 = svmla_f32_z(pred16, C61, B1, A0);
      
      A1=svdup_n_f32(Aptr[baseA + 7]);
      C70 = svmla_f32_z(pred16, C70, B0, A1);
      C71 = svmla_f32_z(pred16, C71, B1, A1);
      
      A0=svdup_n_f32(Aptr[baseA + 8]);
      C80 = svmla_f32_z(pred16, C80, B0, A0);
      C81 = svmla_f32_z(pred16, C81, B1, A0);
      
      A1=svdup_n_f32(Aptr[baseA + 9]);
      C90 = svmla_f32_z(pred16, C90, B0, A1);
      C91 = svmla_f32_z(pred16, C91, B1, A1);

      A0=svdup_n_f32(Aptr[baseA + 10]);
      C100 = svmla_f32_z(pred16, C100, B0, A0);
      C101 = svmla_f32_z(pred16, C101, B1, A0);

      A1=svdup_n_f32(Aptr[baseA + 11]);
      C110 = svmla_f32_z(pred16, C110, B0, A1);
      C111 = svmla_f32_z(pred16, C111, B1, A1);

      A0=svdup_n_f32(Aptr[baseA + 12]);
      C120 = svmla_f32_z(pred16, C120, B0, A0);
      C121 = svmla_f32_z(pred16, C121, B1, A0);

      A1=svdup_n_f32(Aptr[baseA + 13]);
      C130 = svmla_f32_z(pred16, C130, B0, A1);
      C131 = svmla_f32_z(pred16, C131, B1, A1);
      
      baseA = baseA + Amr;
      baseB = baseB + Bnr;
    }
    
    if ( alpha==-one ) {
      C00 = svmul_n_f32_x(pred16, C00, -1.0);
      C10 = svmul_n_f32_x(pred16, C10, -1.0);
      C20 = svmul_n_f32_x(pred16, C20, -1.0);
      C30 = svmul_n_f32_x(pred16, C30, -1.0);
      C40 = svmul_n_f32_x(pred16, C40, -1.0);
      C50 = svmul_n_f32_x(pred16, C50, -1.0);
      C60 = svmul_n_f32_x(pred16, C60, -1.0);
      C70 = svmul_n_f32_x(pred16, C70, -1.0);
      C80 = svmul_n_f32_x(pred16, C80, -1.0);
      C90 = svmul_n_f32_x(pred16, C90, -1.0);
      C100 = svmul_n_f32_x(pred16, C100, -1.0);
      C110 = svmul_n_f32_x(pred16, C110, -1.0);
      C120 = svmul_n_f32_x(pred16, C120, -1.0);
      C130 = svmul_n_f32_x(pred16, C130, -1.0);
      
      C01 = svmul_n_f32_x(pred16, C01, -1.0);
      C11 = svmul_n_f32_x(pred16, C11, -1.0);
      C21 = svmul_n_f32_x(pred16, C21, -1.0);
      C31 = svmul_n_f32_x(pred16, C31, -1.0);
      C41 = svmul_n_f32_x(pred16, C40, -1.0);
      C51 = svmul_n_f32_x(pred16, C51, -1.0);
      C61 = svmul_n_f32_x(pred16, C61, -1.0);
      C71 = svmul_n_f32_x(pred16, C71, -1.0);
      C81 = svmul_n_f32_x(pred16, C81, -1.0);
      C91 = svmul_n_f32_x(pred16, C91, -1.0);
      C101 = svmul_n_f32_x(pred16, C101, -1.0);
      C111 = svmul_n_f32_x(pred16, C111, -1.0);
      C121 = svmul_n_f32_x(pred16, C121, -1.0);
      C131 = svmul_n_f32_x(pred16, C131, -1.0);
    } else if ( alpha!=one ) {
      C00 = svmul_n_f32_x(pred16, C00, alpha);
      C10 = svmul_n_f32_x(pred16, C10, alpha);
      C20 = svmul_n_f32_x(pred16, C20, alpha);
      C30 = svmul_n_f32_x(pred16, C30, alpha);
      C40 = svmul_n_f32_x(pred16, C40, alpha);
      C50 = svmul_n_f32_x(pred16, C50, alpha);
      C60 = svmul_n_f32_x(pred16, C60, alpha);
      C70 = svmul_n_f32_x(pred16, C70, alpha);
      C80 = svmul_n_f32_x(pred16, C80, alpha);
      C90 = svmul_n_f32_x(pred16, C90, alpha);
      C100 = svmul_n_f32_x(pred16, C100, alpha);
      C110 = svmul_n_f32_x(pred16, C110, alpha);
      C120 = svmul_n_f32_x(pred16, C120, alpha);
      C130 = svmul_n_f32_x(pred16, C130, alpha);
      
      C01 = svmul_n_f32_x(pred16, C01, alpha);
      C11 = svmul_n_f32_x(pred16, C11, alpha);
      C21 = svmul_n_f32_x(pred16, C21, alpha);
      C31 = svmul_n_f32_x(pred16, C31, alpha);
      C41 = svmul_n_f32_x(pred16, C41, alpha);
      C51 = svmul_n_f32_x(pred16, C51, alpha);
      C61 = svmul_n_f32_x(pred16, C61, alpha);
      C71 = svmul_n_f32_x(pred16, C71, alpha);
      C81 = svmul_n_f32_x(pred16, C81, alpha);
      C91 = svmul_n_f32_x(pred16, C91, alpha);
      C101 = svmul_n_f32_x(pred16, C101, alpha);
      C111 = svmul_n_f32_x(pred16, C111, alpha);
      C121 = svmul_n_f32_x(pred16, C121, alpha);
      C131 = svmul_n_f32_x(pred16, C131, alpha);

    }
  }
  
  if ( (mr<MR) || (nr<NR) ) {

    svst1_f32(pred16, &Ctrow(0, 0),  C00);
    svst1_f32(pred16, &Ctrow(1, 0),  C10);
    svst1_f32(pred16, &Ctrow(2, 0),  C20);
    svst1_f32(pred16, &Ctrow(3, 0),  C30);
    svst1_f32(pred16, &Ctrow(4, 0),  C40);
    svst1_f32(pred16, &Ctrow(5, 0),  C50);
    svst1_f32(pred16, &Ctrow(6, 0),  C60);
    svst1_f32(pred16, &Ctrow(7, 0),  C70);
    svst1_f32(pred16, &Ctrow(8, 0),  C80);
    svst1_f32(pred16, &Ctrow(9, 0),  C90);
    svst1_f32(pred16, &Ctrow(10, 0),  C100);
    svst1_f32(pred16, &Ctrow(11, 0),  C110);
    svst1_f32(pred16, &Ctrow(12, 0),  C120);
    svst1_f32(pred16, &Ctrow(13, 0),  C130);
    
    svst1_f32(pred16, &Ctrow(0, 16),  C01);
    svst1_f32(pred16, &Ctrow(1, 16),  C11);
    svst1_f32(pred16, &Ctrow(2, 16),  C21);
    svst1_f32(pred16, &Ctrow(3, 16),  C31);
    svst1_f32(pred16, &Ctrow(4, 16),  C41);
    svst1_f32(pred16, &Ctrow(5, 16),  C51);
    svst1_f32(pred16, &Ctrow(6, 16),  C61);
    svst1_f32(pred16, &Ctrow(7, 16),  C71);
    svst1_f32(pred16, &Ctrow(8, 16),  C81);
    svst1_f32(pred16, &Ctrow(9, 16),  C91);
    svst1_f32(pred16, &Ctrow(10, 16),  C101);
    svst1_f32(pred16, &Ctrow(11, 16),  C111);
    svst1_f32(pred16, &Ctrow(12, 16),  C121);
    svst1_f32(pred16, &Ctrow(13, 16),  C131);
    
    if ( beta!=zero ) {
      for ( j=0; j<nr; j++ )
        for ( i=0; i<mr; i++ ) {
          Crow(i,j) = beta*Crow(i,j) + Ctrow(i,j);
  	}
    }
    
    else {
      for ( j=0; j<nr; j++ )
        for ( i=0; i<mr; i++ )
          Crow(i,j) = Ctrow(i,j);
    }
  }
  else if ( (mr==MR) && (nr==NR) ) {
    if (beta != zero) {
      //Set Beta
      Bta=svdup_n_f32(beta);
      
      C0p = svld1_f32(pred16, &Crow(0,0));
      C1p = svld1_f32(pred16, &Crow(1,0));
      C2p = svld1_f32(pred16, &Crow(2,0));
      C3p = svld1_f32(pred16, &Crow(3,0));
      C4p = svld1_f32(pred16, &Crow(4,0));
      C5p = svld1_f32(pred16, &Crow(5,0));
      C6p = svld1_f32(pred16, &Crow(6,0));
      C7p = svld1_f32(pred16, &Crow(7,0));
      C8p = svld1_f32(pred16, &Crow(8,0));
      C9p = svld1_f32(pred16, &Crow(9,0));
      C10p = svld1_f32(pred16, &Crow(10,0));
      C11p = svld1_f32(pred16, &Crow(11,0));
      C12p = svld1_f32(pred16, &Crow(12,0));
      C13p = svld1_f32(pred16, &Crow(13,0));
      
      C00 = svmla_f32_z(pred16, C00, C0p, Bta);
      C10 = svmla_f32_z(pred16, C10, C1p, Bta);
      C20 = svmla_f32_z(pred16, C20, C2p, Bta);
      C30 = svmla_f32_z(pred16, C30, C3p, Bta);
      C40 = svmla_f32_z(pred16, C40, C4p, Bta);
      C50 = svmla_f32_z(pred16, C50, C5p, Bta);
      C60 = svmla_f32_z(pred16, C60, C6p, Bta);
      C70 = svmla_f32_z(pred16, C70, C7p, Bta);
      C80 = svmla_f32_z(pred16, C80, C8p, Bta);
      C90 = svmla_f32_z(pred16, C90, C9p, Bta);
      C100 = svmla_f32_z(pred16, C100, C10p, Bta);
      C110 = svmla_f32_z(pred16, C110, C11p, Bta);
      C120 = svmla_f32_z(pred16, C120, C12p, Bta);
      C130 = svmla_f32_z(pred16, C130, C13p, Bta);
            
      C0p = svld1_f32(pred16, &Crow(0,16));
      C1p = svld1_f32(pred16, &Crow(1,16));
      C2p = svld1_f32(pred16, &Crow(2,16));
      C3p = svld1_f32(pred16, &Crow(3,16));
      C4p = svld1_f32(pred16, &Crow(4,16));
      C5p = svld1_f32(pred16, &Crow(5,16));
      C6p = svld1_f32(pred16, &Crow(6,16));
      C7p = svld1_f32(pred16, &Crow(7,16));
      C8p = svld1_f32(pred16, &Crow(8,16));
      C9p = svld1_f32(pred16, &Crow(9,16));
      C10p = svld1_f32(pred16, &Crow(10,16));
      C11p = svld1_f32(pred16, &Crow(11,16));
      C12p = svld1_f32(pred16, &Crow(12,16));
      C13p = svld1_f32(pred16, &Crow(13,16));
	    
      C01 = svmla_f32_z(pred16, C01, C0p, Bta);
      C11 = svmla_f32_z(pred16, C11, C1p, Bta);
      C21 = svmla_f32_z(pred16, C21, C2p, Bta);
      C31 = svmla_f32_z(pred16, C31, C3p, Bta);
      C41 = svmla_f32_z(pred16, C41, C4p, Bta);
      C51 = svmla_f32_z(pred16, C51, C5p, Bta);
      C61 = svmla_f32_z(pred16, C61, C6p, Bta);
      C71 = svmla_f32_z(pred16, C71, C7p, Bta);
      C81 = svmla_f32_z(pred16, C81, C8p, Bta);
      C91 = svmla_f32_z(pred16, C91, C9p, Bta);
      C101 = svmla_f32_z(pred16, C101, C10p, Bta);
      C111 = svmla_f32_z(pred16, C111, C11p, Bta);
      C121 = svmla_f32_z(pred16, C121, C12p, Bta);
      C131 = svmla_f32_z(pred16, C131, C13p, Bta);
    }
    
    svst1_f32(pred16, &Crow(0, 0), C00);
    svst1_f32(pred16, &Crow(1, 0), C10);
    svst1_f32(pred16, &Crow(2, 0), C20);
    svst1_f32(pred16, &Crow(3, 0), C30);
    svst1_f32(pred16, &Crow(4, 0), C40);
    svst1_f32(pred16, &Crow(5, 0), C50);
    svst1_f32(pred16, &Crow(6, 0), C60);
    svst1_f32(pred16, &Crow(7, 0), C70);
    svst1_f32(pred16, &Crow(8, 0), C80);
    svst1_f32(pred16, &Crow(9, 0), C90);
    svst1_f32(pred16, &Crow(10, 0), C100);
    svst1_f32(pred16, &Crow(11, 0), C110);
    svst1_f32(pred16, &Crow(12, 0), C120);
    svst1_f32(pred16, &Crow(13, 0), C130);

    svst1_f32(pred16, &Crow(0, 16), C01);
    svst1_f32(pred16, &Crow(1, 16), C11);
    svst1_f32(pred16, &Crow(2, 16), C21);
    svst1_f32(pred16, &Crow(3, 16), C31);
    svst1_f32(pred16, &Crow(4, 16), C41);
    svst1_f32(pred16, &Crow(5, 16), C51);
    svst1_f32(pred16, &Crow(6, 16), C61);
    svst1_f32(pred16, &Crow(7, 16), C71);
    svst1_f32(pred16, &Crow(8, 16), C81);
    svst1_f32(pred16, &Crow(9, 16), C91);
    svst1_f32(pred16, &Crow(10, 16), C101);
    svst1_f32(pred16, &Crow(11, 16), C111);
    svst1_f32(pred16, &Crow(12, 16), C121);
    svst1_f32(pred16, &Crow(13, 16), C131);
  }
  else {
    printf("Error: Incorrect use of 4x16 micro-kernel with %d x %d block\n", mr, nr);
    exit(-1);
  }
}

void gemm_microkernel_neon_sve_fixed_16x32_fp32( int mr, int nr, int kc, float alpha, float *Ar, float *Br, float beta, float *C, int ldC ) {
/*
  BLIS GEMM microkernel, computes the product Cr := Cr + Ar * Br
  Specific: only for MRxNR = 4x16
*/
  //SET_MR_NR(15, 32);
  
  int         i, j, k, baseB = 0, baseA = 0, ldCt = NR, Amr, Bnr;

  //Registers for C = (32)
  svfloat32_t C00, C10, C20, C30, C40, C50, C60, C70, C80, C90, C100, C110, C120, C130, C140, C150,
              C01, C11, C21, C31, C41, C51, C61, C71, C81, C91, C101, C111, C121, C131, C141, C151;

  //Other auxiliar Registers (11)
  svfloat32_t  B0,  B1,  A0, A1; //Bta, C0n, C1n;  

  //svfloat32_t C2n, C3n, C4n, C5n, C6n, C7n, C8n, C9n, C10n, C11n, C12n, C13n;
  
  #define C0n B0
  #define C1n B1
  #define Bta A0
  
  float   zero = 0.0, one = 1.0, *Aptr, *Bptr, Ctmp[MR * NR];
  svbool_t pred16 = svwhilelt_b32_u32(0, 16);
  
  if ( kc==0 )
    return;
    
  //Set to 0
  C00=svdup_n_f32(0); C01=svdup_n_f32(0); 
  C10=svdup_n_f32(0); C11=svdup_n_f32(0); 
  C20=svdup_n_f32(0); C21=svdup_n_f32(0); 
  C30=svdup_n_f32(0); C31=svdup_n_f32(0); 
  C40=svdup_n_f32(0); C41=svdup_n_f32(0); 
  C50=svdup_n_f32(0); C51=svdup_n_f32(0); 
  C60=svdup_n_f32(0); C61=svdup_n_f32(0); 
  C70=svdup_n_f32(0); C71=svdup_n_f32(0);
  C80=svdup_n_f32(0); C81=svdup_n_f32(0);
  C90=svdup_n_f32(0); C91=svdup_n_f32(0);
  C100=svdup_n_f32(0); C101=svdup_n_f32(0); 
  C110=svdup_n_f32(0); C111=svdup_n_f32(0);
  C120=svdup_n_f32(0); C121=svdup_n_f32(0); 
  C130=svdup_n_f32(0); C131=svdup_n_f32(0);
  C140=svdup_n_f32(0); C141=svdup_n_f32(0); 
  C150=svdup_n_f32(0); C151=svdup_n_f32(0);
  
  Aptr = &Ar[0];
  Amr  = MR;
  Bptr = &Br[0];
  Bnr  = NR;
  
  if ( alpha!=zero ) {
    for ( k=0; k<kc; k++ ) {
      // Load colums/rows of A/B for current iteration
      B0 = svld1_f32(pred16, &Bptr[baseB]);
      B1 = svld1_f32(pred16, &Bptr[baseB + 16]);
      
      A0=svdup_n_f32(Aptr[baseA + 0]);
      C00 = svmla_f32_z(pred16, C00, B0, A0);
      C01 = svmla_f32_z(pred16, C01, B1, A0);
      
      A1=svdup_n_f32(Aptr[baseA + 1]);
      C10 = svmla_f32_z(pred16, C10, B0, A1);
      C11 = svmla_f32_z(pred16, C11, B1, A1);
      
      A0=svdup_n_f32(Aptr[baseA + 2]);
      C20 = svmla_f32_z(pred16, C20, B0, A0);
      C21 = svmla_f32_z(pred16, C21, B1, A0);
      
      A1=svdup_n_f32(Aptr[baseA + 3]);
      C30 = svmla_f32_z(pred16, C30, B0, A1);
      C31 = svmla_f32_z(pred16, C31, B1, A1);
      
      A0=svdup_n_f32(Aptr[baseA + 4]);
      C40 = svmla_f32_z(pred16, C40, B0, A0);
      C41 = svmla_f32_z(pred16, C41, B1, A0);
      
      A1=svdup_n_f32(Aptr[baseA + 5]);
      C50 = svmla_f32_z(pred16, C50, B0, A1);
      C51 = svmla_f32_z(pred16, C51, B1, A1);
      
      A0=svdup_n_f32(Aptr[baseA + 6]);
      C60 = svmla_f32_z(pred16, C60, B0, A0);
      C61 = svmla_f32_z(pred16, C61, B1, A0);
      
      A1=svdup_n_f32(Aptr[baseA + 7]);
      C70 = svmla_f32_z(pred16, C70, B0, A1);
      C71 = svmla_f32_z(pred16, C71, B1, A1);
      
      A0=svdup_n_f32(Aptr[baseA + 8]);
      C80 = svmla_f32_z(pred16, C80, B0, A0);
      C81 = svmla_f32_z(pred16, C81, B1, A0);
      
      A1=svdup_n_f32(Aptr[baseA + 9]);
      C90 = svmla_f32_z(pred16, C90, B0, A1);
      C91 = svmla_f32_z(pred16, C91, B1, A1);

      A0=svdup_n_f32(Aptr[baseA + 10]);
      C100 = svmla_f32_z(pred16, C100, B0, A0);
      C101 = svmla_f32_z(pred16, C101, B1, A0);

      A1=svdup_n_f32(Aptr[baseA + 11]);
      C110 = svmla_f32_z(pred16, C110, B0, A1);
      C111 = svmla_f32_z(pred16, C111, B1, A1);

      A0=svdup_n_f32(Aptr[baseA + 12]);
      C120 = svmla_f32_z(pred16, C120, B0, A0);
      C121 = svmla_f32_z(pred16, C121, B1, A0);

      A1=svdup_n_f32(Aptr[baseA + 13]);
      C130 = svmla_f32_z(pred16, C130, B0, A1);
      C131 = svmla_f32_z(pred16, C131, B1, A1);

      A0=svdup_n_f32(Aptr[baseA + 14]);
      C140 = svmla_f32_z(pred16, C140, B0, A0);
      C141 = svmla_f32_z(pred16, C141, B1, A0);

      A1=svdup_n_f32(Aptr[baseA + 15]);
      C150 = svmla_f32_z(pred16, C150, B0, A1);
      C151 = svmla_f32_z(pred16, C151, B1, A1);

      baseA = baseA + Amr;
      baseB = baseB + Bnr;
    }
    
  }
  
  if ( (mr<MR) || (nr<NR) ) {
    svst1_f32(pred16, &Ctrow(0, 0),  C00);
    svst1_f32(pred16, &Ctrow(1, 0),  C10);
    svst1_f32(pred16, &Ctrow(2, 0),  C20);
    svst1_f32(pred16, &Ctrow(3, 0),  C30);
    svst1_f32(pred16, &Ctrow(4, 0),  C40);
    svst1_f32(pred16, &Ctrow(5, 0),  C50);
    svst1_f32(pred16, &Ctrow(6, 0),  C60);
    svst1_f32(pred16, &Ctrow(7, 0),  C70);
    svst1_f32(pred16, &Ctrow(8, 0),  C80);
    svst1_f32(pred16, &Ctrow(9, 0),  C90);
    svst1_f32(pred16, &Ctrow(10, 0),  C100);
    svst1_f32(pred16, &Ctrow(11, 0),  C110);
    svst1_f32(pred16, &Ctrow(12, 0),  C120);
    svst1_f32(pred16, &Ctrow(13, 0),  C130);
    svst1_f32(pred16, &Ctrow(14, 0),  C140);
    svst1_f32(pred16, &Ctrow(15, 0),  C150);
    
    svst1_f32(pred16, &Ctrow(0, 16),  C01);
    svst1_f32(pred16, &Ctrow(1, 16),  C11);
    svst1_f32(pred16, &Ctrow(2, 16),  C21);
    svst1_f32(pred16, &Ctrow(3, 16),  C31);
    svst1_f32(pred16, &Ctrow(4, 16),  C41);
    svst1_f32(pred16, &Ctrow(5, 16),  C51);
    svst1_f32(pred16, &Ctrow(6, 16),  C61);
    svst1_f32(pred16, &Ctrow(7, 16),  C71);
    svst1_f32(pred16, &Ctrow(8, 16),  C81);
    svst1_f32(pred16, &Ctrow(9, 16),  C91);
    svst1_f32(pred16, &Ctrow(10, 16),  C101);
    svst1_f32(pred16, &Ctrow(11, 16),  C111);
    svst1_f32(pred16, &Ctrow(12, 16),  C121);
    svst1_f32(pred16, &Ctrow(13, 16),  C131);
    svst1_f32(pred16, &Ctrow(14, 16),  C141);
    svst1_f32(pred16, &Ctrow(15, 16),  C151);
    
    if ( beta!=zero ) {
      for ( j=0; j<nr; j++ )
        for ( i=0; i<mr; i++ ) {
          Crow(i,j) = beta*Crow(i,j) + Ctrow(i,j);
  	}
    }
    
    else {
      for ( j=0; j<nr; j++ )
        for ( i=0; i<mr; i++ )
          Crow(i,j) = Ctrow(i,j);
    }
  }
  else if ( (mr==MR) && (nr==NR) ) {
    if (beta != zero) {
      //Set Beta
      Bta=svdup_n_f32(beta);

      C0n = svld1_f32(pred16, &Crow(0,0));
      C00 = svmla_f32_z(pred16, C00, C0n, Bta);

      C1n = svld1_f32(pred16, &Crow(1,0));
      C10 = svmla_f32_z(pred16, C10, C1n, Bta);

      C0n = svld1_f32(pred16, &Crow(2,0));
      C20 = svmla_f32_z(pred16, C20, C0n, Bta);

      C1n = svld1_f32(pred16, &Crow(3,0));
      C30 = svmla_f32_z(pred16, C30, C1n, Bta);

      C0n = svld1_f32(pred16, &Crow(4,0));
      C40 = svmla_f32_z(pred16, C40, C0n, Bta);

      C1n = svld1_f32(pred16, &Crow(5,0));
      C50 = svmla_f32_z(pred16, C50, C1n, Bta);

      C0n = svld1_f32(pred16, &Crow(6,0));
      C60 = svmla_f32_z(pred16, C60, C0n, Bta);

      C1n = svld1_f32(pred16, &Crow(7,0));
      C70 = svmla_f32_z(pred16, C70, C1n, Bta);

      C0n = svld1_f32(pred16, &Crow(8,0));
      C80 = svmla_f32_z(pred16, C80, C0n, Bta);

      C1n = svld1_f32(pred16, &Crow(9,0));
      C90 = svmla_f32_z(pred16, C90, C1n, Bta);

      C0n = svld1_f32(pred16, &Crow(10,0));
      C100 = svmla_f32_z(pred16, C100, C0n, Bta);

      C1n = svld1_f32(pred16, &Crow(11,0));
      C110 = svmla_f32_z(pred16, C110, C1n, Bta);

      C0n = svld1_f32(pred16, &Crow(12,0));
      C120 = svmla_f32_z(pred16, C120, C0n, Bta);

      C1n = svld1_f32(pred16, &Crow(13,0));
      C130 = svmla_f32_z(pred16, C130, C1n, Bta);

      C0n = svld1_f32(pred16, &Crow(14,0));
      C140 = svmla_f32_z(pred16, C140, C0n, Bta);

      C1n = svld1_f32(pred16, &Crow(15,0));
      C150 = svmla_f32_z(pred16, C150, C1n, Bta);

      

      C0n = svld1_f32(pred16, &Crow(0, 16));
      C01 = svmla_f32_z(pred16, C01, C0n, Bta);

      C1n = svld1_f32(pred16, &Crow(1, 16));
      C11 = svmla_f32_z(pred16, C11, C1n, Bta);

      C0n = svld1_f32(pred16, &Crow(2, 16));
      C21 = svmla_f32_z(pred16, C21, C0n, Bta);

      C1n = svld1_f32(pred16, &Crow(3, 16));
      C31 = svmla_f32_z(pred16, C31, C1n, Bta);

      C0n = svld1_f32(pred16, &Crow(4, 16));
      C41 = svmla_f32_z(pred16, C41, C0n, Bta);

      C1n = svld1_f32(pred16, &Crow(5, 16));
      C51 = svmla_f32_z(pred16, C51, C1n, Bta);

      C0n = svld1_f32(pred16, &Crow(6, 16));
      C61 = svmla_f32_z(pred16, C61, C0n, Bta);

      C1n = svld1_f32(pred16, &Crow(7, 16));
      C71 = svmla_f32_z(pred16, C71, C1n, Bta);

      C0n = svld1_f32(pred16, &Crow(8, 16));
      C81 = svmla_f32_z(pred16, C81, C0n, Bta);

      C1n = svld1_f32(pred16, &Crow(9, 16));
      C91 = svmla_f32_z(pred16, C91, C1n, Bta);

      C0n = svld1_f32(pred16, &Crow(10, 16));
      C101 = svmla_f32_z(pred16, C101, C0n, Bta);

      C1n = svld1_f32(pred16, &Crow(11, 16));
      C111 = svmla_f32_z(pred16, C111, C1n, Bta);

      C0n = svld1_f32(pred16, &Crow(12, 16));
      C121 = svmla_f32_z(pred16, C121, C0n, Bta);

      C1n = svld1_f32(pred16, &Crow(13, 16));
      C131 = svmla_f32_z(pred16, C131, C1n, Bta);

      C0n = svld1_f32(pred16, &Crow(14, 16));
      C141 = svmla_f32_z(pred16, C141, C0n, Bta);

      C1n = svld1_f32(pred16, &Crow(15, 16));
      C151 = svmla_f32_z(pred16, C151, C1n, Bta);

    }
    /* if (beta != zero) { */
    /*   //Set Beta */
    /*   Bta=svdup_n_f32(beta); */
      
    /*   C0n = svld1_f32(pred16, &Crow(0,0)); */
    /*   C1n = svld1_f32(pred16, &Crow(1,0)); */
    /*   C2n = svld1_f32(pred16, &Crow(2,0)); */
    /*   C3n = svld1_f32(pred16, &Crow(3,0)); */
    /*   C4n = svld1_f32(pred16, &Crow(4,0)); */
    /*   C5n = svld1_f32(pred16, &Crow(5,0)); */
    /*   C6n = svld1_f32(pred16, &Crow(6,0)); */
    /*   C7n = svld1_f32(pred16, &Crow(7,0)); */
    /*   C8n = svld1_f32(pred16, &Crow(8,0)); */
    /*   C9n = svld1_f32(pred16, &Crow(9,0)); */
    /*   C10n = svld1_f32(pred16, &Crow(10,0)); */
    /*   C11n = svld1_f32(pred16, &Crow(11,0)); */
    /*   C12n = svld1_f32(pred16, &Crow(12,0)); */
    /*   C13n = svld1_f32(pred16, &Crow(13,0)); */
      
    /*   C00 = svmla_f32_z(pred16, C00, C0n, Bta); */
    /*   C10 = svmla_f32_z(pred16, C10, C1n, Bta); */
    /*   C20 = svmla_f32_z(pred16, C20, C2n, Bta); */
    /*   C30 = svmla_f32_z(pred16, C30, C3n, Bta); */
    /*   C40 = svmla_f32_z(pred16, C40, C4n, Bta); */
    /*   C50 = svmla_f32_z(pred16, C50, C5n, Bta); */
    /*   C60 = svmla_f32_z(pred16, C60, C6n, Bta); */
    /*   C70 = svmla_f32_z(pred16, C70, C7n, Bta); */
    /*   C80 = svmla_f32_z(pred16, C80, C8n, Bta); */
    /*   C90 = svmla_f32_z(pred16, C90, C9n, Bta); */
    /*   C100 = svmla_f32_z(pred16, C100, C10n, Bta); */
    /*   C110 = svmla_f32_z(pred16, C110, C11n, Bta); */
    /*   C120 = svmla_f32_z(pred16, C120, C12n, Bta); */
    /*   C130 = svmla_f32_z(pred16, C130, C13n, Bta); */
            
    /*   C0n = svld1_f32(pred16, &Crow(0,16)); */
    /*   C1n = svld1_f32(pred16, &Crow(1,16)); */
    /*   C2n = svld1_f32(pred16, &Crow(2,16)); */
    /*   C3n = svld1_f32(pred16, &Crow(3,16)); */
    /*   C4n = svld1_f32(pred16, &Crow(4,16)); */
    /*   C5n = svld1_f32(pred16, &Crow(5,16)); */
    /*   C6n = svld1_f32(pred16, &Crow(6,16)); */
    /*   C7n = svld1_f32(pred16, &Crow(7,16)); */
    /*   C8n = svld1_f32(pred16, &Crow(8,16)); */
    /*   C9n = svld1_f32(pred16, &Crow(9,16)); */
    /*   C10n = svld1_f32(pred16, &Crow(10,16)); */
    /*   C11n = svld1_f32(pred16, &Crow(11,16)); */
    /*   C12n = svld1_f32(pred16, &Crow(12,16)); */
    /*   C13n = svld1_f32(pred16, &Crow(13,16)); */
	    
    /*   C01 = svmla_f32_z(pred16, C01, C0n, Bta); */
    /*   C11 = svmla_f32_z(pred16, C11, C1n, Bta); */
    /*   C21 = svmla_f32_z(pred16, C21, C2n, Bta); */
    /*   C31 = svmla_f32_z(pred16, C31, C3n, Bta); */
    /*   C41 = svmla_f32_z(pred16, C41, C4n, Bta); */
    /*   C51 = svmla_f32_z(pred16, C51, C5n, Bta); */
    /*   C61 = svmla_f32_z(pred16, C61, C6n, Bta); */
    /*   C71 = svmla_f32_z(pred16, C71, C7n, Bta); */
    /*   C81 = svmla_f32_z(pred16, C81, C8n, Bta); */
    /*   C91 = svmla_f32_z(pred16, C91, C9n, Bta); */
    /*   C101 = svmla_f32_z(pred16, C101, C10n, Bta); */
    /*   C111 = svmla_f32_z(pred16, C111, C11n, Bta); */
    /*   C121 = svmla_f32_z(pred16, C121, C12n, Bta); */
    /*   C131 = svmla_f32_z(pred16, C131, C13n, Bta); */
    /* } */
    
    svst1_f32(pred16, &Crow(0, 0), C00);
    svst1_f32(pred16, &Crow(1, 0), C10);
    svst1_f32(pred16, &Crow(2, 0), C20);
    svst1_f32(pred16, &Crow(3, 0), C30);
    svst1_f32(pred16, &Crow(4, 0), C40);
    svst1_f32(pred16, &Crow(5, 0), C50);
    svst1_f32(pred16, &Crow(6, 0), C60);
    svst1_f32(pred16, &Crow(7, 0), C70);
    svst1_f32(pred16, &Crow(8, 0), C80);
    svst1_f32(pred16, &Crow(9, 0), C90);
    svst1_f32(pred16, &Crow(10, 0), C100);
    svst1_f32(pred16, &Crow(11, 0), C110);
    svst1_f32(pred16, &Crow(12, 0), C120);
    svst1_f32(pred16, &Crow(13, 0), C130);
    svst1_f32(pred16, &Crow(14, 0), C140);
    svst1_f32(pred16, &Crow(15, 0), C150);

    svst1_f32(pred16, &Crow(0, 16), C01);
    svst1_f32(pred16, &Crow(1, 16), C11);
    svst1_f32(pred16, &Crow(2, 16), C21);
    svst1_f32(pred16, &Crow(3, 16), C31);
    svst1_f32(pred16, &Crow(4, 16), C41);
    svst1_f32(pred16, &Crow(5, 16), C51);
    svst1_f32(pred16, &Crow(6, 16), C61);
    svst1_f32(pred16, &Crow(7, 16), C71);
    svst1_f32(pred16, &Crow(8, 16), C81);
    svst1_f32(pred16, &Crow(9, 16), C91);
    svst1_f32(pred16, &Crow(10, 16), C101);
    svst1_f32(pred16, &Crow(11, 16), C111);
    svst1_f32(pred16, &Crow(12, 16), C121);
    svst1_f32(pred16, &Crow(13, 16), C131);
    svst1_f32(pred16, &Crow(14, 16), C141);
    svst1_f32(pred16, &Crow(15, 16), C151);
  }
  else {
    printf("Error: Incorrect use of 4x16 micro-kernel with %d x %d block\n", mr, nr);
    exit(-1);
  }
}

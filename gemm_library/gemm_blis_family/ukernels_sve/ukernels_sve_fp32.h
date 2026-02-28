/* 
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

#include <stdio.h>
#include <stdlib.h>
//#include <arm_neon.h>
#include <arm_sve.h>
#include "../qblis.h"

//-----------------------------------------------------------------------------------------------------------
//  Micro-kernels for BLIS
//-----------------------------------------------------------------------------------------------------------

void gemm_microkernel_sve_asm_32x10_fp32( int, float *, float *, float *, int, float, float);
void gemm_microkernel_sve_32x10_fp32( char, int, int, int, float, float *, float *, float, float *, int);


//-----------------------------------------------------------------------------------------------------------

void gemm_microkernel_neon_sve_fixed_16x32_fp32(int, int, int, float, float *, float *, float, float *, int);

void gemm_microkernel_neon_sve_fixed_14x32_fp32(int, int, int, float, float *, float *, float, float *, int);

void gemm_microkernel_neon_sve_fixed_12x32_fp32(int, int, int, float, float *, float *, float, float *, int);

void gemm_microkernel_neon_sve_10x32_fp32(int, int, int, float, float *, float *, float, float *, int);

void gemm_microkernel_neon_sve_fixed_4x32_fp32(int, int, int, float, float *, float *, float, float *, int);

void gemm_microkernel_neon_sve_fixed_4x16_fp32(int, int, int, float, float *, float *, float, float *, int);

void gemm_microkernel_neon_sve_fixed_4x4_fp32(int, int, int, float, float *, float *, float, float *, int);






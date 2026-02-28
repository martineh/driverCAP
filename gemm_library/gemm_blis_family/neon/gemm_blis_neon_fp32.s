	.arch armv8-a
	.file	"gemm_blis_neon_fp32.c"
	.text
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align	3
.LC0:
	.string	"Error: Incorrect use of 4x4 micro-kernel with %d x %d block\n"
	.text
	.align	2
	.p2align 4,,11
	.global	gemm_microkernel_Cresident_neon_4x4_fp32
	.type	gemm_microkernel_Cresident_neon_4x4_fp32, %function
gemm_microkernel_Cresident_neon_4x4_fp32:
.LFB4307:
	.cfi_startproc
	stp	x29, x30, [sp, -144]!
	.cfi_def_cfa_offset 144
	.cfi_offset 29, -144
	.cfi_offset 30, -136
	adrp	x8, :got:__stack_chk_guard
	mov	x29, sp
	ldr	x8, [x8, #:got_lo12:__stack_chk_guard]
	ldr	x9, [x8]
	str	x9, [sp, 136]
	mov	x9, 0
	cbz	w3, .L1
	and	w0, w0, 255
	stp	x19, x20, [sp, 16]
	.cfi_offset 20, -120
	.cfi_offset 19, -128
	cmp	w0, 67
	csel	x8, x4, x5, eq
	csel	x4, x5, x4, eq
	fcmp	s0, #0.0
	mov	w20, w2
	movi	v4.4s, 0
	beq	.L37
	cmp	w3, 0
	ble	.L38
	mov	v5.16b, v4.16b
	mov	x5, x8
	mov	v6.16b, v4.16b
	add	x8, x8, x3, uxtw 4
	mov	v7.16b, v4.16b
	.p2align 3,,7
.L6:
	ldr	q2, [x5], 16
	ldr	q3, [x4], 16
	fmla	v7.4s, v2.4s, v3.s[0]
	fmla	v6.4s, v2.4s, v3.s[1]
	fmla	v5.4s, v2.4s, v3.s[2]
	fmla	v4.4s, v2.4s, v3.s[3]
	cmp	x5, x8
	bne	.L6
.L5:
	fmov	s2, -1.0e+0
	fcmp	s0, s2
	beq	.L97
	fmov	s2, 1.0e+0
	fcmp	s0, s2
	beq	.L4
	dup	v0.4s, v0.s[0]
	fmul	v7.4s, v7.4s, v0.4s
	fmul	v6.4s, v6.4s, v0.4s
	fmul	v5.4s, v5.4s, v0.4s
	fmul	v4.4s, v4.4s, v0.4s
	b	.L4
	.p2align 2,,3
.L8:
	cmp	w1, 4
	ccmp	w20, 4, 0, eq
	bne	.L33
	fcmp	s1, #0.0
	sbfiz	x7, x7, 2, 32
	add	x0, x6, x7
	add	x1, x0, x7
	beq	.L34
	ldr	q3, [x6, x7]
	ldr	q2, [x0, x7]
	ldr	q0, [x1, x7]
	ldr	q16, [x6]
	fmla	v6.4s, v3.4s, v1.s[0]
	fmla	v5.4s, v2.4s, v1.s[0]
	fmla	v4.4s, v0.4s, v1.s[0]
	fmla	v7.4s, v16.4s, v1.s[0]
.L34:
	str	q7, [x6]
	str	q6, [x6, x7]
	str	q5, [x0, x7]
	str	q4, [x1, x7]
	ldp	x19, x20, [sp, 16]
	.cfi_restore 20
	.cfi_restore 19
.L1:
	adrp	x0, :got:__stack_chk_guard
	ldr	x0, [x0, #:got_lo12:__stack_chk_guard]
	ldr	x1, [sp, 136]
	ldr	x2, [x0]
	subs	x1, x1, x2
	mov	x2, 0
	bne	.L98
	ldp	x29, x30, [sp], 144
	.cfi_restore 30
	.cfi_restore 29
	.cfi_def_cfa_offset 0
	ret
	.p2align 2,,3
.L37:
	.cfi_def_cfa_offset 144
	.cfi_offset 19, -128
	.cfi_offset 20, -120
	.cfi_offset 29, -144
	.cfi_offset 30, -136
	mov	v5.16b, v4.16b
	mov	v6.16b, v4.16b
	mov	v7.16b, v4.16b
.L4:
	cmp	w1, 3
	ccmp	w20, 3, 4, gt
	bgt	.L8
	fcmp	s1, #0.0
	stp	q7, q6, [sp, 64]
	add	x19, sp, 64
	stp	q5, q4, [sp, 96]
	beq	.L9
	cmp	w0, 67
	beq	.L10
	cmp	w20, 0
	ble	.L93
	cmp	w1, 0
	ble	.L93
	dup	v2.4s, v1.s[0]
	sub	w11, w20, #1
	lsr	w9, w20, 2
	and	w10, w20, -4
	and	w12, w20, 3
	mov	w8, 0
	mov	w5, 0
	.p2align 3,,7
.L13:
	cmp	w11, 2
	bls	.L99
	sbfiz	x3, x5, 2, 32
	sbfiz	x0, x8, 2, 32
	add	x2, x6, x3
	add	x4, x19, x0
	ldr	q3, [x6, x3]
	ldr	q0, [x19, x0]
	fmla	v0.4s, v2.4s, v3.4s
	str	q0, [x6, x3]
	cmp	w9, 1
	bls	.L20
	ldr	q3, [x2, 16]
	ldr	q0, [x4, 16]
	fmla	v0.4s, v2.4s, v3.4s
	str	q0, [x2, 16]
	cmp	w9, 2
	beq	.L20
	ldr	q3, [x2, 32]
	ldr	q0, [x4, 32]
	fmla	v0.4s, v2.4s, v3.4s
	str	q0, [x2, 32]
	cmp	w9, 3
	beq	.L20
	ldr	q3, [x2, 48]
	ldr	q0, [x4, 48]
	fmla	v0.4s, v3.4s, v2.4s
	str	q0, [x2, 48]
.L20:
	cmp	w10, w20
	beq	.L21
	mov	w13, w12
	mov	w0, w10
.L22:
	add	w2, w0, w5
	add	w4, w0, w8
	add	w3, w0, 1
	sxtw	x2, w2
	ldr	s0, [x19, w4, sxtw 2]
	ldr	s3, [x6, x2, lsl 2]
	fmadd	s0, s1, s3, s0
	str	s0, [x6, x2, lsl 2]
	cmp	w13, 1
	beq	.L21
	add	w2, w5, w3
	add	w3, w8, w3
	add	w0, w0, 2
	sxtw	x2, w2
	ldr	s0, [x19, w3, sxtw 2]
	ldr	s3, [x6, x2, lsl 2]
	fmadd	s0, s1, s3, s0
	str	s0, [x6, x2, lsl 2]
	cmp	w13, 2
	beq	.L21
	add	w2, w5, w0
	add	w0, w8, w0
	sxtw	x2, w2
	ldr	s0, [x19, w0, sxtw 2]
	ldr	s3, [x6, x2, lsl 2]
	fmadd	s0, s1, s3, s0
	str	s0, [x6, x2, lsl 2]
.L21:
	add	w5, w5, w7
	add	w8, w8, 4
	subs	w1, w1, #1
	bne	.L13
.L93:
	ldp	x19, x20, [sp, 16]
	.cfi_remember_state
	.cfi_restore 20
	.cfi_restore 19
	b	.L1
	.p2align 2,,3
.L97:
	.cfi_restore_state
	fneg	v7.4s, v7.4s
	fneg	v6.4s, v6.4s
	fneg	v5.4s, v5.4s
	fneg	v4.4s, v4.4s
	b	.L4
	.p2align 2,,3
.L9:
	cmp	w0, 67
	beq	.L24
	cmp	w20, 0
	ble	.L93
	cmp	w1, 0
	ble	.L93
	sub	w12, w20, #1
	lsr	w9, w20, 2
	and	w10, w20, -4
	and	w13, w20, 3
	mov	w3, 0
	mov	w2, 0
	.p2align 3,,7
.L26:
	cmp	w12, 2
	bls	.L100
	sbfiz	x0, x2, 2, 32
	sbfiz	x4, x3, 2, 32
	add	x8, x19, x0
	add	x5, x6, x4
	ldr	q0, [x19, x0]
	str	q0, [x6, x4]
	cmp	w9, 1
	bls	.L29
	ldr	q0, [x8, 16]
	str	q0, [x5, 16]
	cmp	w9, 2
	beq	.L29
	ldr	q0, [x8, 32]
	str	q0, [x5, 32]
	cmp	w9, 3
	beq	.L29
	ldr	q0, [x8, 48]
	str	q0, [x5, 48]
.L29:
	cmp	w20, w10
	beq	.L30
	mov	w8, w13
	mov	w0, w10
.L31:
	add	w5, w0, w2
	add	w11, w0, w3
	add	w4, w0, 1
	ldr	s0, [x19, w5, sxtw 2]
	str	s0, [x6, w11, sxtw 2]
	cmp	w8, 1
	beq	.L30
	add	w5, w2, w4
	add	w4, w3, w4
	add	w0, w0, 2
	ldr	s0, [x19, w5, sxtw 2]
	str	s0, [x6, w4, sxtw 2]
	cmp	w8, 2
	beq	.L30
	add	w4, w2, w0
	add	w0, w3, w0
	ldr	s0, [x19, w4, sxtw 2]
	str	s0, [x6, w0, sxtw 2]
.L30:
	add	w2, w2, 4
	add	w3, w3, w7
	subs	w1, w1, #1
	bne	.L26
	ldp	x19, x20, [sp, 16]
	.cfi_remember_state
	.cfi_restore 20
	.cfi_restore 19
	b	.L1
	.p2align 2,,3
.L100:
	.cfi_restore_state
	mov	w8, w20
	mov	w0, 0
	b	.L31
	.p2align 2,,3
.L99:
	mov	w13, w20
	mov	w0, 0
	b	.L22
	.p2align 2,,3
.L24:
	cmp	w20, 0
	ble	.L93
	cmp	w1, 0
	ble	.L93
	stp	x21, x22, [sp, 32]
	.cfi_offset 22, -104
	.cfi_offset 21, -112
	sbfiz	x22, x1, 2, 32
	mov	w21, 0
	str	x23, [sp, 48]
	.cfi_offset 23, -96
	sbfiz	x23, x7, 2, 32
.L28:
	mov	x1, x19
	mov	x0, x6
	mov	x2, x22
	add	w21, w21, 1
	bl	memcpy
	add	x19, x19, 16
	add	x6, x0, x23
	cmp	w20, w21
	bne	.L28
	ldp	x19, x20, [sp, 16]
	.cfi_restore 20
	.cfi_restore 19
	ldp	x21, x22, [sp, 32]
	.cfi_restore 22
	.cfi_restore 21
	ldr	x23, [sp, 48]
	.cfi_restore 23
	b	.L1
	.p2align 2,,3
.L10:
	.cfi_offset 19, -128
	.cfi_offset 20, -120
	cmp	w20, 0
	ble	.L93
	cmp	w1, 0
	ble	.L93
	dup	v2.4s, v1.s[0]
	lsl	w20, w20, 2
	sbfiz	x14, x7, 2, 32
	mov	x2, x6
	mov	x5, x19
	sub	w13, w1, #1
	lsr	w11, w1, 2
	and	w12, w1, -4
	mov	w4, 0
	mov	w9, 0
.L15:
	cmp	w13, 2
	bls	.L101
	ldr	q3, [x2]
	ldr	q0, [x5]
	fmla	v0.4s, v3.4s, v2.4s
	str	q0, [x2]
	cmp	w11, 1
	bls	.L16
	ldr	q3, [x2, 16]
	ldr	q0, [x5, 16]
	fmla	v0.4s, v2.4s, v3.4s
	str	q0, [x2, 16]
	cmp	w11, 2
	beq	.L16
	ldr	q3, [x2, 32]
	ldr	q0, [x5, 32]
	fmla	v0.4s, v2.4s, v3.4s
	str	q0, [x2, 32]
	cmp	w11, 3
	beq	.L16
	ldr	q3, [x2, 48]
	ldr	q0, [x5, 48]
	fmla	v0.4s, v2.4s, v3.4s
	str	q0, [x2, 48]
.L16:
	mov	w0, w12
	cmp	w1, w12
	beq	.L17
.L18:
	add	w3, w9, w0
	add	w10, w4, w0
	add	w8, w0, 1
	sxtw	x3, w3
	ldr	s0, [x19, w10, sxtw 2]
	ldr	s3, [x6, x3, lsl 2]
	fmadd	s0, s3, s1, s0
	str	s0, [x6, x3, lsl 2]
	cmp	w1, w8
	ble	.L17
	add	w3, w9, w8
	add	w8, w4, w8
	add	w0, w0, 2
	sxtw	x3, w3
	ldr	s0, [x19, w8, sxtw 2]
	ldr	s3, [x6, x3, lsl 2]
	fmadd	s0, s3, s1, s0
	str	s0, [x6, x3, lsl 2]
	cmp	w0, w1
	bge	.L17
	add	w3, w9, w0
	add	w0, w4, w0
	sxtw	x3, w3
	ldr	s0, [x19, w0, sxtw 2]
	ldr	s3, [x6, x3, lsl 2]
	fmadd	s0, s1, s3, s0
	str	s0, [x6, x3, lsl 2]
.L17:
	add	w4, w4, 4
	add	w9, w9, w7
	add	x2, x2, x14
	add	x5, x5, 16
	cmp	w20, w4
	bne	.L15
	ldp	x19, x20, [sp, 16]
	.cfi_remember_state
	.cfi_restore 20
	.cfi_restore 19
	b	.L1
.L38:
	.cfi_restore_state
	mov	v5.16b, v4.16b
	mov	v6.16b, v4.16b
	mov	v7.16b, v4.16b
	b	.L5
.L101:
	mov	w0, 0
	b	.L18
.L98:
	.cfi_restore 19
	.cfi_restore 20
	stp	x19, x20, [sp, 16]
	.cfi_offset 20, -120
	.cfi_offset 19, -128
	stp	x21, x22, [sp, 32]
	.cfi_offset 22, -104
	.cfi_offset 21, -112
	str	x23, [sp, 48]
	.cfi_offset 23, -96
	bl	__stack_chk_fail
.L33:
	.cfi_restore 21
	.cfi_restore 22
	.cfi_restore 23
	mov	w2, w1
	mov	w3, w20
	adrp	x1, .LC0
	add	x1, x1, :lo12:.LC0
	mov	w0, 1
	stp	x21, x22, [sp, 32]
	.cfi_offset 22, -104
	.cfi_offset 21, -112
	str	x23, [sp, 48]
	.cfi_offset 23, -96
	bl	__printf_chk
	mov	w0, -1
	bl	exit
	.cfi_endproc
.LFE4307:
	.size	gemm_microkernel_Cresident_neon_4x4_fp32, .-gemm_microkernel_Cresident_neon_4x4_fp32
	.align	2
	.p2align 4,,11
	.global	gemm_microkernel_Cresident_neon_4x4_prefetch_fp32
	.type	gemm_microkernel_Cresident_neon_4x4_prefetch_fp32, %function
gemm_microkernel_Cresident_neon_4x4_prefetch_fp32:
.LFB4308:
	.cfi_startproc
	stp	x29, x30, [sp, -144]!
	.cfi_def_cfa_offset 144
	.cfi_offset 29, -144
	.cfi_offset 30, -136
	adrp	x8, :got:__stack_chk_guard
	mov	x29, sp
	ldr	x8, [x8, #:got_lo12:__stack_chk_guard]
	ldr	x9, [x8]
	str	x9, [sp, 136]
	mov	x9, 0
	cbz	w3, .L102
	and	w0, w0, 255
	stp	x19, x20, [sp, 16]
	.cfi_offset 20, -120
	.cfi_offset 19, -128
	cmp	w0, 67
	movi	v4.4s, 0
	mov	w20, w2
	csel	x2, x4, x5, eq
	csel	x5, x5, x4, eq
	fcmp	s0, #0.0
	ldr	q16, [x2]
	ldr	q17, [x5]
	beq	.L138
	cmp	w3, 1
	ble	.L139
	sub	w3, w3, #2
	mov	x8, 32
	mov	v5.16b, v4.16b
	add	x4, x5, 16
	add	x3, x8, x3, uxtw 4
	add	x5, x2, 16
	mov	v6.16b, v4.16b
	add	x3, x2, x3
	mov	v7.16b, v4.16b
	.p2align 3,,7
.L107:
	mov	v2.16b, v16.16b
	mov	v3.16b, v17.16b
	ldr	q16, [x5], 16
	ldr	q17, [x4], 16
	fmla	v7.4s, v2.4s, v3.s[0]
	fmla	v6.4s, v2.4s, v3.s[1]
	fmla	v5.4s, v2.4s, v3.s[2]
	fmla	v4.4s, v2.4s, v3.s[3]
	cmp	x5, x3
	bne	.L107
.L106:
	fmla	v7.4s, v16.4s, v17.s[0]
	fmov	s2, -1.0e+0
	fmla	v6.4s, v16.4s, v17.s[1]
	fmla	v5.4s, v16.4s, v17.s[2]
	fcmp	s0, s2
	fmla	v4.4s, v16.4s, v17.s[3]
	beq	.L197
	fmov	s2, 1.0e+0
	fcmp	s0, s2
	beq	.L105
	dup	v0.4s, v0.s[0]
	fmul	v7.4s, v7.4s, v0.4s
	fmul	v6.4s, v6.4s, v0.4s
	fmul	v5.4s, v5.4s, v0.4s
	fmul	v4.4s, v4.4s, v0.4s
	b	.L105
	.p2align 2,,3
.L109:
	cmp	w1, 4
	ccmp	w20, 4, 0, eq
	bne	.L134
	fcmp	s1, #0.0
	sbfiz	x7, x7, 2, 32
	add	x0, x6, x7
	add	x1, x0, x7
	beq	.L135
	ldr	q3, [x6, x7]
	ldr	q2, [x0, x7]
	ldr	q0, [x1, x7]
	ldr	q16, [x6]
	fmla	v6.4s, v3.4s, v1.s[0]
	fmla	v5.4s, v2.4s, v1.s[0]
	fmla	v4.4s, v0.4s, v1.s[0]
	fmla	v7.4s, v16.4s, v1.s[0]
.L135:
	str	q7, [x6]
	str	q6, [x6, x7]
	str	q5, [x0, x7]
	str	q4, [x1, x7]
	ldp	x19, x20, [sp, 16]
	.cfi_restore 20
	.cfi_restore 19
.L102:
	adrp	x0, :got:__stack_chk_guard
	ldr	x0, [x0, #:got_lo12:__stack_chk_guard]
	ldr	x1, [sp, 136]
	ldr	x2, [x0]
	subs	x1, x1, x2
	mov	x2, 0
	bne	.L198
	ldp	x29, x30, [sp], 144
	.cfi_restore 30
	.cfi_restore 29
	.cfi_def_cfa_offset 0
	ret
	.p2align 2,,3
.L138:
	.cfi_def_cfa_offset 144
	.cfi_offset 19, -128
	.cfi_offset 20, -120
	.cfi_offset 29, -144
	.cfi_offset 30, -136
	mov	v5.16b, v4.16b
	mov	v6.16b, v4.16b
	mov	v7.16b, v4.16b
.L105:
	cmp	w1, 3
	ccmp	w20, 3, 4, gt
	bgt	.L109
	fcmp	s1, #0.0
	stp	q7, q6, [sp, 64]
	add	x19, sp, 64
	stp	q5, q4, [sp, 96]
	beq	.L110
	cmp	w0, 67
	beq	.L111
	cmp	w20, 0
	ble	.L194
	cmp	w1, 0
	ble	.L194
	dup	v2.4s, v1.s[0]
	sub	w11, w20, #1
	lsr	w9, w20, 2
	and	w10, w20, -4
	and	w12, w20, 3
	mov	w8, 0
	mov	w5, 0
	.p2align 3,,7
.L114:
	cmp	w11, 2
	bls	.L199
	sbfiz	x3, x5, 2, 32
	sbfiz	x0, x8, 2, 32
	add	x2, x6, x3
	add	x4, x19, x0
	ldr	q3, [x6, x3]
	ldr	q0, [x19, x0]
	fmla	v0.4s, v2.4s, v3.4s
	str	q0, [x6, x3]
	cmp	w9, 1
	bls	.L121
	ldr	q3, [x2, 16]
	ldr	q0, [x4, 16]
	fmla	v0.4s, v2.4s, v3.4s
	str	q0, [x2, 16]
	cmp	w9, 2
	beq	.L121
	ldr	q3, [x2, 32]
	ldr	q0, [x4, 32]
	fmla	v0.4s, v2.4s, v3.4s
	str	q0, [x2, 32]
	cmp	w9, 3
	beq	.L121
	ldr	q3, [x2, 48]
	ldr	q0, [x4, 48]
	fmla	v0.4s, v3.4s, v2.4s
	str	q0, [x2, 48]
.L121:
	cmp	w10, w20
	beq	.L122
	mov	w13, w12
	mov	w0, w10
.L123:
	add	w2, w0, w5
	add	w4, w0, w8
	add	w3, w0, 1
	sxtw	x2, w2
	ldr	s0, [x19, w4, sxtw 2]
	ldr	s3, [x6, x2, lsl 2]
	fmadd	s0, s1, s3, s0
	str	s0, [x6, x2, lsl 2]
	cmp	w13, 1
	beq	.L122
	add	w2, w5, w3
	add	w3, w8, w3
	add	w0, w0, 2
	sxtw	x2, w2
	ldr	s0, [x19, w3, sxtw 2]
	ldr	s3, [x6, x2, lsl 2]
	fmadd	s0, s1, s3, s0
	str	s0, [x6, x2, lsl 2]
	cmp	w13, 2
	beq	.L122
	add	w2, w5, w0
	add	w0, w8, w0
	sxtw	x2, w2
	ldr	s0, [x19, w0, sxtw 2]
	ldr	s3, [x6, x2, lsl 2]
	fmadd	s0, s1, s3, s0
	str	s0, [x6, x2, lsl 2]
.L122:
	add	w5, w5, w7
	add	w8, w8, 4
	subs	w1, w1, #1
	bne	.L114
.L194:
	ldp	x19, x20, [sp, 16]
	.cfi_remember_state
	.cfi_restore 20
	.cfi_restore 19
	b	.L102
	.p2align 2,,3
.L197:
	.cfi_restore_state
	fneg	v7.4s, v7.4s
	fneg	v6.4s, v6.4s
	fneg	v5.4s, v5.4s
	fneg	v4.4s, v4.4s
	b	.L105
	.p2align 2,,3
.L110:
	cmp	w0, 67
	beq	.L125
	cmp	w20, 0
	ble	.L194
	cmp	w1, 0
	ble	.L194
	sub	w12, w20, #1
	lsr	w9, w20, 2
	and	w10, w20, -4
	and	w13, w20, 3
	mov	w3, 0
	mov	w2, 0
	.p2align 3,,7
.L127:
	cmp	w12, 2
	bls	.L200
	sbfiz	x0, x2, 2, 32
	sbfiz	x4, x3, 2, 32
	add	x8, x19, x0
	add	x5, x6, x4
	ldr	q0, [x19, x0]
	str	q0, [x6, x4]
	cmp	w9, 1
	bls	.L130
	ldr	q0, [x8, 16]
	str	q0, [x5, 16]
	cmp	w9, 2
	beq	.L130
	ldr	q0, [x8, 32]
	str	q0, [x5, 32]
	cmp	w9, 3
	beq	.L130
	ldr	q0, [x8, 48]
	str	q0, [x5, 48]
.L130:
	cmp	w20, w10
	beq	.L131
	mov	w8, w13
	mov	w0, w10
.L132:
	add	w5, w0, w2
	add	w11, w0, w3
	add	w4, w0, 1
	ldr	s0, [x19, w5, sxtw 2]
	str	s0, [x6, w11, sxtw 2]
	cmp	w8, 1
	beq	.L131
	add	w5, w2, w4
	add	w4, w3, w4
	add	w0, w0, 2
	ldr	s0, [x19, w5, sxtw 2]
	str	s0, [x6, w4, sxtw 2]
	cmp	w8, 2
	beq	.L131
	add	w4, w2, w0
	add	w0, w3, w0
	ldr	s0, [x19, w4, sxtw 2]
	str	s0, [x6, w0, sxtw 2]
.L131:
	add	w2, w2, 4
	add	w3, w3, w7
	subs	w1, w1, #1
	bne	.L127
	ldp	x19, x20, [sp, 16]
	.cfi_remember_state
	.cfi_restore 20
	.cfi_restore 19
	b	.L102
	.p2align 2,,3
.L200:
	.cfi_restore_state
	mov	w8, w20
	mov	w0, 0
	b	.L132
	.p2align 2,,3
.L199:
	mov	w13, w20
	mov	w0, 0
	b	.L123
	.p2align 2,,3
.L125:
	cmp	w20, 0
	ble	.L194
	cmp	w1, 0
	ble	.L194
	stp	x21, x22, [sp, 32]
	.cfi_offset 22, -104
	.cfi_offset 21, -112
	sbfiz	x22, x1, 2, 32
	mov	w21, 0
	str	x23, [sp, 48]
	.cfi_offset 23, -96
	sbfiz	x23, x7, 2, 32
.L129:
	mov	x1, x19
	mov	x0, x6
	mov	x2, x22
	add	w21, w21, 1
	bl	memcpy
	add	x19, x19, 16
	add	x6, x0, x23
	cmp	w20, w21
	bne	.L129
	ldp	x19, x20, [sp, 16]
	.cfi_restore 20
	.cfi_restore 19
	ldp	x21, x22, [sp, 32]
	.cfi_restore 22
	.cfi_restore 21
	ldr	x23, [sp, 48]
	.cfi_restore 23
	b	.L102
	.p2align 2,,3
.L111:
	.cfi_offset 19, -128
	.cfi_offset 20, -120
	cmp	w20, 0
	ble	.L194
	cmp	w1, 0
	ble	.L194
	dup	v2.4s, v1.s[0]
	lsl	w20, w20, 2
	sbfiz	x14, x7, 2, 32
	mov	x2, x6
	mov	x5, x19
	sub	w13, w1, #1
	lsr	w11, w1, 2
	and	w12, w1, -4
	mov	w4, 0
	mov	w9, 0
.L116:
	cmp	w13, 2
	bls	.L201
	ldr	q3, [x2]
	ldr	q0, [x5]
	fmla	v0.4s, v3.4s, v2.4s
	str	q0, [x2]
	cmp	w11, 1
	bls	.L117
	ldr	q3, [x2, 16]
	ldr	q0, [x5, 16]
	fmla	v0.4s, v2.4s, v3.4s
	str	q0, [x2, 16]
	cmp	w11, 2
	beq	.L117
	ldr	q3, [x2, 32]
	ldr	q0, [x5, 32]
	fmla	v0.4s, v2.4s, v3.4s
	str	q0, [x2, 32]
	cmp	w11, 3
	beq	.L117
	ldr	q3, [x2, 48]
	ldr	q0, [x5, 48]
	fmla	v0.4s, v2.4s, v3.4s
	str	q0, [x2, 48]
.L117:
	mov	w0, w12
	cmp	w12, w1
	beq	.L118
.L119:
	add	w3, w9, w0
	add	w10, w4, w0
	add	w8, w0, 1
	sxtw	x3, w3
	ldr	s0, [x19, w10, sxtw 2]
	ldr	s3, [x6, x3, lsl 2]
	fmadd	s0, s3, s1, s0
	str	s0, [x6, x3, lsl 2]
	cmp	w1, w8
	ble	.L118
	add	w3, w9, w8
	add	w8, w4, w8
	add	w0, w0, 2
	sxtw	x3, w3
	ldr	s0, [x19, w8, sxtw 2]
	ldr	s3, [x6, x3, lsl 2]
	fmadd	s0, s3, s1, s0
	str	s0, [x6, x3, lsl 2]
	cmp	w0, w1
	bge	.L118
	add	w3, w9, w0
	add	w0, w4, w0
	sxtw	x3, w3
	ldr	s0, [x19, w0, sxtw 2]
	ldr	s3, [x6, x3, lsl 2]
	fmadd	s0, s1, s3, s0
	str	s0, [x6, x3, lsl 2]
.L118:
	add	w4, w4, 4
	add	w9, w9, w7
	add	x2, x2, x14
	add	x5, x5, 16
	cmp	w20, w4
	bne	.L116
	ldp	x19, x20, [sp, 16]
	.cfi_remember_state
	.cfi_restore 20
	.cfi_restore 19
	b	.L102
.L139:
	.cfi_restore_state
	mov	v5.16b, v4.16b
	mov	v6.16b, v4.16b
	mov	v7.16b, v4.16b
	b	.L106
.L201:
	mov	w0, 0
	b	.L119
.L198:
	.cfi_restore 19
	.cfi_restore 20
	stp	x19, x20, [sp, 16]
	.cfi_offset 20, -120
	.cfi_offset 19, -128
	stp	x21, x22, [sp, 32]
	.cfi_offset 22, -104
	.cfi_offset 21, -112
	str	x23, [sp, 48]
	.cfi_offset 23, -96
	bl	__stack_chk_fail
.L134:
	.cfi_restore 21
	.cfi_restore 22
	.cfi_restore 23
	mov	w2, w1
	mov	w3, w20
	adrp	x1, .LC0
	add	x1, x1, :lo12:.LC0
	mov	w0, 1
	stp	x21, x22, [sp, 32]
	.cfi_offset 22, -104
	.cfi_offset 21, -112
	str	x23, [sp, 48]
	.cfi_offset 23, -96
	bl	__printf_chk
	mov	w0, -1
	bl	exit
	.cfi_endproc
.LFE4308:
	.size	gemm_microkernel_Cresident_neon_4x4_prefetch_fp32, .-gemm_microkernel_Cresident_neon_4x4_prefetch_fp32
	.align	2
	.p2align 4,,11
	.global	gemm_microkernel_Cresident_neon_4x4_prefetch_unroll_fp32
	.type	gemm_microkernel_Cresident_neon_4x4_prefetch_unroll_fp32, %function
gemm_microkernel_Cresident_neon_4x4_prefetch_unroll_fp32:
.LFB4309:
	.cfi_startproc
	stp	x29, x30, [sp, -144]!
	.cfi_def_cfa_offset 144
	.cfi_offset 29, -144
	.cfi_offset 30, -136
	adrp	x8, :got:__stack_chk_guard
	mov	x29, sp
	ldr	x8, [x8, #:got_lo12:__stack_chk_guard]
	ldr	x9, [x8]
	str	x9, [sp, 136]
	mov	x9, 0
	cbz	w3, .L202
	and	w0, w0, 255
	stp	x19, x20, [sp, 16]
	.cfi_offset 20, -120
	.cfi_offset 19, -128
	cmp	w0, 67
	csel	x10, x4, x5, eq
	mov	w19, w2
	csel	x2, x5, x4, eq
	fcmp	s0, #0.0
	ldr	q4, [x10]
	ldr	q2, [x2]
	beq	.L240
	cmp	w3, 2
	ble	.L241
	sub	w9, w3, #3
	add	x4, x10, 16
	movi	v3.4s, 0
	add	x5, x2, 32
	lsr	w9, w9, 1
	ubfiz	x8, x9, 5, 31
	mov	v5.16b, v3.16b
	add	x8, x8, 48
	mov	v6.16b, v3.16b
	add	x8, x10, x8
	mov	v7.16b, v3.16b
	.p2align 3,,7
.L207:
	fmla	v7.4s, v4.4s, v2.s[0]
	fmla	v6.4s, v4.4s, v2.s[1]
	fmla	v5.4s, v4.4s, v2.s[2]
	fmla	v3.4s, v4.4s, v2.s[3]
	ldr	q16, [x5, -16]
	ldp	q2, q4, [x4]
	add	x4, x4, 32
	fmla	v7.4s, v2.4s, v16.s[0]
	fmla	v6.4s, v2.4s, v16.s[1]
	fmla	v5.4s, v2.4s, v16.s[2]
	fmla	v3.4s, v2.4s, v16.s[3]
	ldr	q2, [x5], 32
	cmp	x8, x4
	bne	.L207
	add	w9, w9, 1
	lsl	w9, w9, 3
.L206:
	dup	v18.4s, v2.s[0]
	dup	v17.4s, v2.s[1]
	dup	v16.4s, v2.s[2]
	dup	v2.4s, v2.s[3]
	tbz	x3, 0, .L299
	fmla	v3.4s, v4.4s, v2.4s
	fmov	s2, -1.0e+0
	fmla	v7.4s, v4.4s, v18.4s
	fmla	v6.4s, v4.4s, v17.4s
	fcmp	s0, s2
	fmla	v5.4s, v4.4s, v16.4s
	bne	.L209
.L303:
	fneg	v7.4s, v7.4s
	cmp	w1, 3
	fneg	v6.4s, v6.4s
	ccmp	w19, 3, 4, gt
	fneg	v5.4s, v5.4s
	fneg	v3.4s, v3.4s
	ble	.L300
.L210:
	cmp	w1, 4
	ccmp	w19, 4, 0, eq
	bne	.L235
	fcmp	s1, #0.0
	sbfiz	x7, x7, 2, 32
	add	x0, x6, x7
	add	x1, x0, x7
	beq	.L236
	ldr	q4, [x6, x7]
	ldr	q2, [x0, x7]
	ldr	q0, [x1, x7]
	ldr	q16, [x6]
	fmla	v6.4s, v4.4s, v1.s[0]
	fmla	v5.4s, v2.4s, v1.s[0]
	fmla	v3.4s, v0.4s, v1.s[0]
	fmla	v7.4s, v16.4s, v1.s[0]
.L236:
	str	q7, [x6]
	str	q6, [x6, x7]
	str	q5, [x0, x7]
	str	q3, [x1, x7]
	ldp	x19, x20, [sp, 16]
	.cfi_restore 20
	.cfi_restore 19
.L202:
	adrp	x0, :got:__stack_chk_guard
	ldr	x0, [x0, #:got_lo12:__stack_chk_guard]
	ldr	x1, [sp, 136]
	ldr	x2, [x0]
	subs	x1, x1, x2
	mov	x2, 0
	bne	.L301
	ldp	x29, x30, [sp], 144
	.cfi_restore 30
	.cfi_restore 29
	.cfi_def_cfa_offset 0
	ret
	.p2align 2,,3
.L240:
	.cfi_def_cfa_offset 144
	.cfi_offset 19, -128
	.cfi_offset 20, -120
	.cfi_offset 29, -144
	.cfi_offset 30, -136
	movi	v3.4s, 0
	mov	v5.16b, v3.16b
	mov	v6.16b, v3.16b
	mov	v7.16b, v3.16b
.L205:
	cmp	w1, 3
	ccmp	w19, 3, 4, gt
	bgt	.L210
.L300:
	fcmp	s1, #0.0
	stp	q7, q6, [sp, 64]
	add	x20, sp, 64
	stp	q5, q3, [sp, 96]
	beq	.L211
	cmp	w0, 67
	beq	.L212
	cmp	w19, 0
	ble	.L296
	cmp	w1, 0
	ble	.L296
	dup	v2.4s, v1.s[0]
	sub	w11, w19, #1
	lsr	w9, w19, 2
	and	w10, w19, -4
	and	w12, w19, 3
	mov	w4, 0
	mov	w3, 0
	.p2align 3,,7
.L215:
	cmp	w11, 2
	bls	.L302
	sbfiz	x5, x3, 2, 32
	sbfiz	x0, x4, 2, 32
	add	x2, x6, x5
	add	x8, x20, x0
	ldr	q3, [x6, x5]
	ldr	q0, [x20, x0]
	fmla	v0.4s, v2.4s, v3.4s
	str	q0, [x6, x5]
	cmp	w9, 1
	bls	.L222
	ldr	q3, [x2, 16]
	ldr	q0, [x8, 16]
	fmla	v0.4s, v2.4s, v3.4s
	str	q0, [x2, 16]
	cmp	w9, 2
	beq	.L222
	ldr	q3, [x2, 32]
	ldr	q0, [x8, 32]
	fmla	v0.4s, v2.4s, v3.4s
	str	q0, [x2, 32]
	cmp	w9, 3
	beq	.L222
	ldr	q3, [x2, 48]
	ldr	q0, [x8, 48]
	fmla	v0.4s, v3.4s, v2.4s
	str	q0, [x2, 48]
.L222:
	cmp	w10, w19
	beq	.L223
	mov	w13, w12
	mov	w0, w10
.L224:
	add	w2, w0, w3
	add	w8, w0, w4
	add	w5, w0, 1
	sxtw	x2, w2
	ldr	s0, [x20, w8, sxtw 2]
	ldr	s3, [x6, x2, lsl 2]
	fmadd	s0, s1, s3, s0
	str	s0, [x6, x2, lsl 2]
	cmp	w13, 1
	beq	.L223
	add	w2, w3, w5
	add	w5, w4, w5
	add	w0, w0, 2
	sxtw	x2, w2
	ldr	s0, [x20, w5, sxtw 2]
	ldr	s3, [x6, x2, lsl 2]
	fmadd	s0, s1, s3, s0
	str	s0, [x6, x2, lsl 2]
	cmp	w13, 2
	beq	.L223
	add	w2, w3, w0
	add	w0, w4, w0
	sxtw	x2, w2
	ldr	s0, [x20, w0, sxtw 2]
	ldr	s3, [x6, x2, lsl 2]
	fmadd	s0, s1, s3, s0
	str	s0, [x6, x2, lsl 2]
.L223:
	add	w3, w3, w7
	add	w4, w4, 4
	subs	w1, w1, #1
	bne	.L215
.L296:
	ldp	x19, x20, [sp, 16]
	.cfi_remember_state
	.cfi_restore 20
	.cfi_restore 19
	b	.L202
	.p2align 2,,3
.L299:
	.cfi_restore_state
	fmla	v3.4s, v4.4s, v2.4s
	add	w9, w9, 4
	fmla	v7.4s, v4.4s, v18.4s
	fmla	v6.4s, v4.4s, v17.4s
	sbfiz	x3, x9, 2, 32
	fmla	v5.4s, v4.4s, v16.4s
	ldr	q2, [x10, x3]
	ldr	q4, [x2, x3]
	fmla	v7.4s, v2.4s, v4.s[0]
	fmla	v6.4s, v2.4s, v4.s[1]
	fmla	v5.4s, v2.4s, v4.s[2]
	fmla	v3.4s, v2.4s, v4.s[3]
	fmov	s2, -1.0e+0
	fcmp	s0, s2
	beq	.L303
.L209:
	fmov	s2, 1.0e+0
	fcmp	s0, s2
	beq	.L205
	dup	v0.4s, v0.s[0]
	fmul	v7.4s, v7.4s, v0.4s
	fmul	v6.4s, v6.4s, v0.4s
	fmul	v5.4s, v5.4s, v0.4s
	fmul	v3.4s, v3.4s, v0.4s
	b	.L205
	.p2align 2,,3
.L211:
	cmp	w0, 67
	beq	.L226
	cmp	w19, 0
	ble	.L296
	cmp	w1, 0
	ble	.L296
	sub	w12, w19, #1
	lsr	w9, w19, 2
	and	w10, w19, -4
	and	w13, w19, 3
	mov	w3, 0
	mov	w2, 0
	.p2align 3,,7
.L228:
	cmp	w12, 2
	bls	.L304
	sbfiz	x0, x2, 2, 32
	sbfiz	x4, x3, 2, 32
	add	x8, x20, x0
	add	x5, x6, x4
	ldr	q0, [x20, x0]
	str	q0, [x6, x4]
	cmp	w9, 1
	bls	.L231
	ldr	q0, [x8, 16]
	str	q0, [x5, 16]
	cmp	w9, 2
	beq	.L231
	ldr	q0, [x8, 32]
	str	q0, [x5, 32]
	cmp	w9, 3
	beq	.L231
	ldr	q0, [x8, 48]
	str	q0, [x5, 48]
.L231:
	cmp	w19, w10
	beq	.L232
	mov	w8, w13
	mov	w0, w10
.L233:
	add	w5, w0, w2
	add	w11, w0, w3
	add	w4, w0, 1
	ldr	s0, [x20, w5, sxtw 2]
	str	s0, [x6, w11, sxtw 2]
	cmp	w8, 1
	beq	.L232
	add	w5, w2, w4
	add	w4, w3, w4
	add	w0, w0, 2
	ldr	s0, [x20, w5, sxtw 2]
	str	s0, [x6, w4, sxtw 2]
	cmp	w8, 2
	beq	.L232
	add	w4, w2, w0
	add	w0, w3, w0
	ldr	s0, [x20, w4, sxtw 2]
	str	s0, [x6, w0, sxtw 2]
.L232:
	add	w2, w2, 4
	add	w3, w3, w7
	subs	w1, w1, #1
	bne	.L228
	ldp	x19, x20, [sp, 16]
	.cfi_remember_state
	.cfi_restore 20
	.cfi_restore 19
	b	.L202
	.p2align 2,,3
.L302:
	.cfi_restore_state
	mov	w13, w19
	mov	w0, 0
	b	.L224
	.p2align 2,,3
.L304:
	mov	w8, w19
	mov	w0, 0
	b	.L233
	.p2align 2,,3
.L226:
	cmp	w19, 0
	ble	.L296
	cmp	w1, 0
	ble	.L296
	stp	x21, x22, [sp, 32]
	.cfi_offset 22, -104
	.cfi_offset 21, -112
	sbfiz	x22, x1, 2, 32
	mov	w21, 0
	str	x23, [sp, 48]
	.cfi_offset 23, -96
	sbfiz	x23, x7, 2, 32
.L230:
	mov	x1, x20
	mov	x0, x6
	mov	x2, x22
	add	w21, w21, 1
	bl	memcpy
	add	x20, x20, 16
	add	x6, x0, x23
	cmp	w19, w21
	bne	.L230
	ldp	x19, x20, [sp, 16]
	.cfi_restore 20
	.cfi_restore 19
	ldp	x21, x22, [sp, 32]
	.cfi_restore 22
	.cfi_restore 21
	ldr	x23, [sp, 48]
	.cfi_restore 23
	b	.L202
	.p2align 2,,3
.L212:
	.cfi_offset 19, -128
	.cfi_offset 20, -120
	cmp	w19, 0
	ble	.L296
	cmp	w1, 0
	ble	.L296
	dup	v2.4s, v1.s[0]
	lsl	w19, w19, 2
	sbfiz	x14, x7, 2, 32
	mov	x2, x6
	mov	x5, x20
	sub	w13, w1, #1
	lsr	w11, w1, 2
	and	w12, w1, -4
	mov	w4, 0
	mov	w9, 0
.L217:
	cmp	w13, 2
	bls	.L305
	ldr	q3, [x2]
	ldr	q0, [x5]
	fmla	v0.4s, v3.4s, v2.4s
	str	q0, [x2]
	cmp	w11, 1
	bls	.L218
	ldr	q3, [x2, 16]
	ldr	q0, [x5, 16]
	fmla	v0.4s, v2.4s, v3.4s
	str	q0, [x2, 16]
	cmp	w11, 2
	beq	.L218
	ldr	q3, [x2, 32]
	ldr	q0, [x5, 32]
	fmla	v0.4s, v2.4s, v3.4s
	str	q0, [x2, 32]
	cmp	w11, 3
	beq	.L218
	ldr	q3, [x2, 48]
	ldr	q0, [x5, 48]
	fmla	v0.4s, v2.4s, v3.4s
	str	q0, [x2, 48]
.L218:
	mov	w0, w12
	cmp	w12, w1
	beq	.L219
.L220:
	add	w3, w9, w0
	add	w10, w4, w0
	add	w8, w0, 1
	sxtw	x3, w3
	ldr	s0, [x20, w10, sxtw 2]
	ldr	s3, [x6, x3, lsl 2]
	fmadd	s0, s3, s1, s0
	str	s0, [x6, x3, lsl 2]
	cmp	w1, w8
	ble	.L219
	add	w3, w9, w8
	add	w8, w4, w8
	add	w0, w0, 2
	sxtw	x3, w3
	ldr	s0, [x20, w8, sxtw 2]
	ldr	s3, [x6, x3, lsl 2]
	fmadd	s0, s3, s1, s0
	str	s0, [x6, x3, lsl 2]
	cmp	w0, w1
	bge	.L219
	add	w3, w9, w0
	add	w0, w4, w0
	sxtw	x3, w3
	ldr	s0, [x20, w0, sxtw 2]
	ldr	s3, [x6, x3, lsl 2]
	fmadd	s0, s1, s3, s0
	str	s0, [x6, x3, lsl 2]
.L219:
	add	w4, w4, 4
	add	w9, w9, w7
	add	x2, x2, x14
	add	x5, x5, 16
	cmp	w19, w4
	bne	.L217
	ldp	x19, x20, [sp, 16]
	.cfi_remember_state
	.cfi_restore 20
	.cfi_restore 19
	b	.L202
.L241:
	.cfi_restore_state
	movi	v3.4s, 0
	mov	w9, 0
	mov	v5.16b, v3.16b
	mov	v6.16b, v3.16b
	mov	v7.16b, v3.16b
	b	.L206
.L305:
	mov	w0, 0
	b	.L220
.L301:
	.cfi_restore 19
	.cfi_restore 20
	stp	x19, x20, [sp, 16]
	.cfi_offset 20, -120
	.cfi_offset 19, -128
	stp	x21, x22, [sp, 32]
	.cfi_offset 22, -104
	.cfi_offset 21, -112
	str	x23, [sp, 48]
	.cfi_offset 23, -96
	bl	__stack_chk_fail
.L235:
	.cfi_restore 21
	.cfi_restore 22
	.cfi_restore 23
	mov	w2, w1
	mov	w3, w19
	adrp	x1, .LC0
	add	x1, x1, :lo12:.LC0
	mov	w0, 1
	stp	x21, x22, [sp, 32]
	.cfi_offset 22, -104
	.cfi_offset 21, -112
	str	x23, [sp, 48]
	.cfi_offset 23, -96
	bl	__printf_chk
	mov	w0, -1
	bl	exit
	.cfi_endproc
.LFE4309:
	.size	gemm_microkernel_Cresident_neon_4x4_prefetch_unroll_fp32, .-gemm_microkernel_Cresident_neon_4x4_prefetch_unroll_fp32
	.section	.rodata.str1.8
	.align	3
.LC1:
	.string	"Error: Incorrect use of 8x8 micro-kernel with %d x %d block\n"
	.text
	.align	2
	.p2align 4,,11
	.global	gemm_microkernel_Cresident_neon_8x8_fp32
	.type	gemm_microkernel_Cresident_neon_8x8_fp32, %function
gemm_microkernel_Cresident_neon_8x8_fp32:
.LFB4310:
	.cfi_startproc
	stp	x29, x30, [sp, -144]!
	.cfi_def_cfa_offset 144
	.cfi_offset 29, -144
	.cfi_offset 30, -136
	adrp	x8, :got:__stack_chk_guard
	mov	x29, sp
	ldr	x8, [x8, #:got_lo12:__stack_chk_guard]
	ldr	x9, [x8]
	str	x9, [sp, 136]
	mov	x9, 0
	cbz	w3, .L306
	and	w0, w0, 255
	stp	x19, x20, [sp, 16]
	.cfi_offset 20, -120
	.cfi_offset 19, -128
	cmp	w0, 67
	csel	x8, x4, x5, eq
	csel	x5, x5, x4, eq
	fcmp	s0, #0.0
	mov	w19, w2
	movi	v6.4s, 0
	beq	.L342
	cmp	w3, 0
	ble	.L343
	mov	v7.16b, v6.16b
	add	x9, x8, x3, uxtw 4
	mov	v16.16b, v6.16b
	add	x4, x5, 16
	mov	v17.16b, v6.16b
	mov	v18.16b, v6.16b
	mov	v19.16b, v6.16b
	mov	v20.16b, v6.16b
	mov	v21.16b, v6.16b
	mov	v23.16b, v6.16b
	mov	v24.16b, v6.16b
	mov	v25.16b, v6.16b
	mov	v26.16b, v6.16b
	mov	v27.16b, v6.16b
	mov	v28.16b, v6.16b
	mov	v29.16b, v6.16b
	mov	v22.16b, v6.16b
	ldr	q2, [x8]
	.p2align 3,,7
.L311:
	mov	v5.16b, v2.16b
	ldr	q4, [x4, -16]
	ldr	q2, [x8, 16]!
	ldr	q3, [x4], 16
	fmla	v22.4s, v5.4s, v4.s[0]
	fmla	v29.4s, v5.4s, v4.s[1]
	fmla	v28.4s, v5.4s, v4.s[2]
	fmla	v27.4s, v5.4s, v4.s[3]
	fmla	v26.4s, v5.4s, v3.s[0]
	fmla	v25.4s, v5.4s, v3.s[1]
	fmla	v24.4s, v5.4s, v3.s[2]
	fmla	v23.4s, v5.4s, v3.s[3]
	fmla	v21.4s, v2.4s, v4.s[0]
	fmla	v20.4s, v2.4s, v4.s[1]
	fmla	v19.4s, v2.4s, v4.s[2]
	fmla	v18.4s, v2.4s, v4.s[3]
	fmla	v17.4s, v2.4s, v3.s[0]
	fmla	v16.4s, v2.4s, v3.s[1]
	fmla	v7.4s, v2.4s, v3.s[2]
	fmla	v6.4s, v2.4s, v3.s[3]
	cmp	x8, x9
	bne	.L311
.L310:
	fmov	s2, -1.0e+0
	fcmp	s0, s2
	beq	.L401
	fmov	s2, 1.0e+0
	fcmp	s0, s2
	beq	.L309
	dup	v0.4s, v0.s[0]
	fmul	v22.4s, v22.4s, v0.4s
	fmul	v29.4s, v29.4s, v0.4s
	fmul	v28.4s, v28.4s, v0.4s
	fmul	v27.4s, v27.4s, v0.4s
	fmul	v26.4s, v26.4s, v0.4s
	fmul	v25.4s, v25.4s, v0.4s
	fmul	v24.4s, v24.4s, v0.4s
	fmul	v23.4s, v23.4s, v0.4s
	fmul	v21.4s, v21.4s, v0.4s
	fmul	v20.4s, v20.4s, v0.4s
	fmul	v19.4s, v19.4s, v0.4s
	fmul	v18.4s, v18.4s, v0.4s
	fmul	v17.4s, v17.4s, v0.4s
	fmul	v16.4s, v16.4s, v0.4s
	fmul	v7.4s, v7.4s, v0.4s
	fmul	v6.4s, v6.4s, v0.4s
	b	.L309
	.p2align 2,,3
.L313:
	cmp	w1, 4
	ccmp	w19, 4, 0, eq
	bne	.L338
	lsl	w8, w7, 1
	sbfiz	x1, x7, 2, 32
	add	w0, w8, w7
	lsl	w2, w7, 3
	sub	w2, w2, w7
	add	w5, w7, w7, lsl 2
	lsl	w4, w0, 1
	add	x12, x6, x1
	add	x11, x12, x1
	fcmp	s1, #0.0
	sxtw	x3, w0
	add	x10, x11, x1
	sxtw	x0, w7
	sxtw	x4, w4
	sxtw	x2, w2
	sxtw	x5, w5
	add	x9, x10, x1
	add	x4, x4, 4
	add	x3, x3, 4
	add	x2, x2, 4
	add	x5, x5, 4
	add	x0, x0, 1
	mov	x13, 16
	add	x7, x9, x1
	add	x8, x13, x8, sxtw 2
	add	x14, x7, x1
	lsl	x4, x4, 2
	lsl	x3, x3, 2
	lsl	x2, x2, 2
	lsl	x5, x5, 2
	lsl	x0, x0, 4
	add	x13, x1, x13
	beq	.L339
	str	d8, [sp, 56]
	.cfi_offset 72, -88
	ldr	q8, [x9, x1]
	ldr	q4, [x6, x1]
	fmla	v25.4s, v8.4s, v1.s[0]
	ldr	q8, [x7, x1]
	ldr	q3, [x12, x1]
	fmla	v24.4s, v8.4s, v1.s[0]
	ldr	q8, [x14, x1]
	ldr	q2, [x11, x1]
	fmla	v23.4s, v8.4s, v1.s[0]
	ldr	q0, [x10, x1]
	ldp	q5, q8, [x6]
	fmla	v29.4s, v4.4s, v1.s[0]
	fmla	v28.4s, v3.4s, v1.s[0]
	fmla	v27.4s, v2.4s, v1.s[0]
	fmla	v26.4s, v0.4s, v1.s[0]
	fmla	v22.4s, v5.4s, v1.s[0]
	ldr	q31, [x6, x13]
	ldr	q30, [x6, x8]
	ldr	q4, [x6, x0]
	ldr	q3, [x6, x5]
	ldr	q2, [x6, x4]
	ldr	q0, [x6, x2]
	ldr	q5, [x6, x3]
	fmla	v20.4s, v31.4s, v1.s[0]
	fmla	v19.4s, v30.4s, v1.s[0]
	fmla	v17.4s, v4.4s, v1.s[0]
	fmla	v16.4s, v3.4s, v1.s[0]
	fmla	v7.4s, v2.4s, v1.s[0]
	fmla	v6.4s, v0.4s, v1.s[0]
	fmla	v18.4s, v5.4s, v1.s[0]
	fmla	v21.4s, v8.4s, v1.s[0]
	ldr	d8, [sp, 56]
	.cfi_restore 72
.L339:
	str	q22, [x6]
	str	q29, [x6, x1]
	str	q28, [x12, x1]
	str	q27, [x11, x1]
	str	q26, [x10, x1]
	str	q25, [x9, x1]
	str	q24, [x7, x1]
	str	q23, [x14, x1]
	str	q21, [x6, 16]
	str	q20, [x6, x13]
	str	q19, [x6, x8]
	str	q18, [x6, x3]
	str	q17, [x6, x0]
	str	q16, [x6, x5]
	str	q7, [x6, x4]
	str	q6, [x6, x2]
	ldp	x19, x20, [sp, 16]
	.cfi_restore 20
	.cfi_restore 19
.L306:
	adrp	x0, :got:__stack_chk_guard
	ldr	x0, [x0, #:got_lo12:__stack_chk_guard]
	ldr	x1, [sp, 136]
	ldr	x2, [x0]
	subs	x1, x1, x2
	mov	x2, 0
	bne	.L402
	ldp	x29, x30, [sp], 144
	.cfi_restore 30
	.cfi_restore 29
	.cfi_def_cfa_offset 0
	ret
	.p2align 2,,3
.L342:
	.cfi_def_cfa_offset 144
	.cfi_offset 19, -128
	.cfi_offset 20, -120
	.cfi_offset 29, -144
	.cfi_offset 30, -136
	mov	v7.16b, v6.16b
	mov	v16.16b, v6.16b
	mov	v17.16b, v6.16b
	mov	v18.16b, v6.16b
	mov	v19.16b, v6.16b
	mov	v20.16b, v6.16b
	mov	v21.16b, v6.16b
	mov	v23.16b, v6.16b
	mov	v24.16b, v6.16b
	mov	v25.16b, v6.16b
	mov	v26.16b, v6.16b
	mov	v27.16b, v6.16b
	mov	v28.16b, v6.16b
	mov	v29.16b, v6.16b
	mov	v22.16b, v6.16b
.L309:
	cmp	w1, 3
	ccmp	w19, 3, 4, gt
	bgt	.L313
	fcmp	s1, #0.0
	stp	q22, q21, [sp, 64]
	add	x20, sp, 64
	stp	q20, q19, [sp, 96]
	stp	q18, q17, [sp, 128]
	stp	q16, q7, [sp, 160]
	str	q6, [sp, 192]
	beq	.L314
	cmp	w0, 67
	beq	.L315
	cmp	w19, 0
	ble	.L398
	cmp	w1, 0
	ble	.L398
	dup	v2.4s, v1.s[0]
	sub	w11, w19, #1
	lsr	w9, w19, 2
	and	w10, w19, -4
	and	w12, w19, 3
	mov	w8, 0
	mov	w5, 0
	.p2align 3,,7
.L318:
	cmp	w11, 2
	bls	.L403
	sbfiz	x3, x5, 2, 32
	sbfiz	x0, x8, 2, 32
	add	x2, x6, x3
	add	x4, x20, x0
	ldr	q3, [x6, x3]
	ldr	q0, [x20, x0]
	fmla	v0.4s, v2.4s, v3.4s
	str	q0, [x6, x3]
	cmp	w9, 1
	bls	.L325
	ldr	q3, [x2, 16]
	ldr	q0, [x4, 16]
	fmla	v0.4s, v3.4s, v2.4s
	str	q0, [x2, 16]
	cmp	w9, 2
	beq	.L325
	ldr	q3, [x2, 32]
	ldr	q0, [x4, 32]
	fmla	v0.4s, v3.4s, v2.4s
	str	q0, [x2, 32]
	cmp	w9, 3
	beq	.L325
	ldr	q3, [x2, 48]
	ldr	q0, [x4, 48]
	fmla	v0.4s, v3.4s, v2.4s
	str	q0, [x2, 48]
.L325:
	cmp	w19, w10
	beq	.L326
	mov	w13, w12
	mov	w0, w10
.L327:
	add	w2, w0, w5
	add	w4, w0, w8
	add	w3, w0, 1
	sxtw	x2, w2
	ldr	s0, [x20, w4, sxtw 2]
	ldr	s3, [x6, x2, lsl 2]
	fmadd	s0, s1, s3, s0
	str	s0, [x6, x2, lsl 2]
	cmp	w13, 1
	beq	.L326
	add	w2, w5, w3
	add	w3, w8, w3
	add	w0, w0, 2
	sxtw	x2, w2
	ldr	s0, [x20, w3, sxtw 2]
	ldr	s3, [x6, x2, lsl 2]
	fmadd	s0, s1, s3, s0
	str	s0, [x6, x2, lsl 2]
	cmp	w13, 2
	beq	.L326
	add	w2, w5, w0
	add	w0, w8, w0
	sxtw	x2, w2
	ldr	s0, [x20, w0, sxtw 2]
	ldr	s3, [x6, x2, lsl 2]
	fmadd	s0, s1, s3, s0
	str	s0, [x6, x2, lsl 2]
.L326:
	add	w5, w5, w7
	add	w8, w8, 4
	subs	w1, w1, #1
	bne	.L318
.L398:
	ldp	x19, x20, [sp, 16]
	.cfi_remember_state
	.cfi_restore 20
	.cfi_restore 19
	b	.L306
	.p2align 2,,3
.L401:
	.cfi_restore_state
	fneg	v22.4s, v22.4s
	fneg	v29.4s, v29.4s
	fneg	v28.4s, v28.4s
	fneg	v27.4s, v27.4s
	fneg	v26.4s, v26.4s
	fneg	v25.4s, v25.4s
	fneg	v24.4s, v24.4s
	fneg	v23.4s, v23.4s
	fneg	v21.4s, v21.4s
	fneg	v20.4s, v20.4s
	fneg	v19.4s, v19.4s
	fneg	v18.4s, v18.4s
	fneg	v17.4s, v17.4s
	fneg	v16.4s, v16.4s
	fneg	v7.4s, v7.4s
	fneg	v6.4s, v6.4s
	b	.L309
	.p2align 2,,3
.L314:
	cmp	w0, 67
	beq	.L329
	cmp	w19, 0
	ble	.L398
	cmp	w1, 0
	ble	.L398
	sub	w12, w19, #1
	lsr	w9, w19, 2
	and	w10, w19, -4
	and	w13, w19, 3
	mov	w3, 0
	mov	w2, 0
	.p2align 3,,7
.L331:
	cmp	w12, 2
	bls	.L404
	sbfiz	x0, x2, 2, 32
	sbfiz	x4, x3, 2, 32
	add	x8, x20, x0
	add	x5, x6, x4
	ldr	q0, [x20, x0]
	str	q0, [x6, x4]
	cmp	w9, 1
	bls	.L334
	ldr	q0, [x8, 16]
	str	q0, [x5, 16]
	cmp	w9, 2
	beq	.L334
	ldr	q0, [x8, 32]
	str	q0, [x5, 32]
	cmp	w9, 3
	beq	.L334
	ldr	q0, [x8, 48]
	str	q0, [x5, 48]
.L334:
	cmp	w19, w10
	beq	.L335
	mov	w8, w13
	mov	w0, w10
.L336:
	add	w5, w0, w2
	add	w11, w0, w3
	add	w4, w0, 1
	ldr	s0, [x20, w5, sxtw 2]
	str	s0, [x6, w11, sxtw 2]
	cmp	w8, 1
	beq	.L335
	add	w5, w2, w4
	add	w4, w3, w4
	add	w0, w0, 2
	ldr	s0, [x20, w5, sxtw 2]
	str	s0, [x6, w4, sxtw 2]
	cmp	w8, 2
	beq	.L335
	add	w4, w2, w0
	add	w0, w3, w0
	ldr	s0, [x20, w4, sxtw 2]
	str	s0, [x6, w0, sxtw 2]
.L335:
	add	w2, w2, 4
	add	w3, w3, w7
	subs	w1, w1, #1
	bne	.L331
	ldp	x19, x20, [sp, 16]
	.cfi_remember_state
	.cfi_restore 20
	.cfi_restore 19
	b	.L306
	.p2align 2,,3
.L404:
	.cfi_restore_state
	mov	w8, w19
	mov	w0, 0
	b	.L336
	.p2align 2,,3
.L403:
	mov	w13, w19
	mov	w0, 0
	b	.L327
	.p2align 2,,3
.L329:
	cmp	w19, 0
	ble	.L398
	cmp	w1, 0
	ble	.L398
	stp	x21, x22, [sp, 32]
	.cfi_offset 22, -104
	.cfi_offset 21, -112
	sbfiz	x22, x1, 2, 32
	mov	w21, 0
	str	x23, [sp, 48]
	.cfi_offset 23, -96
	sbfiz	x23, x7, 2, 32
.L333:
	mov	x1, x20
	mov	x0, x6
	mov	x2, x22
	add	w21, w21, 1
	bl	memcpy
	add	x20, x20, 16
	add	x6, x0, x23
	cmp	w19, w21
	bne	.L333
	ldp	x19, x20, [sp, 16]
	.cfi_restore 20
	.cfi_restore 19
	ldp	x21, x22, [sp, 32]
	.cfi_restore 22
	.cfi_restore 21
	ldr	x23, [sp, 48]
	.cfi_restore 23
	b	.L306
	.p2align 2,,3
.L315:
	.cfi_offset 19, -128
	.cfi_offset 20, -120
	cmp	w19, 0
	ble	.L398
	cmp	w1, 0
	ble	.L398
	dup	v2.4s, v1.s[0]
	lsl	w19, w19, 2
	sbfiz	x14, x7, 2, 32
	mov	x2, x6
	mov	x5, x20
	sub	w13, w1, #1
	lsr	w11, w1, 2
	and	w12, w1, -4
	mov	w4, 0
	mov	w9, 0
.L320:
	cmp	w13, 2
	bls	.L405
	ldr	q3, [x2]
	ldr	q0, [x5]
	fmla	v0.4s, v3.4s, v2.4s
	str	q0, [x2]
	cmp	w11, 1
	bls	.L321
	ldr	q3, [x2, 16]
	ldr	q0, [x5, 16]
	fmla	v0.4s, v2.4s, v3.4s
	str	q0, [x2, 16]
	cmp	w11, 2
	beq	.L321
	ldr	q3, [x2, 32]
	ldr	q0, [x5, 32]
	fmla	v0.4s, v2.4s, v3.4s
	str	q0, [x2, 32]
	cmp	w11, 3
	beq	.L321
	ldr	q3, [x2, 48]
	ldr	q0, [x5, 48]
	fmla	v0.4s, v2.4s, v3.4s
	str	q0, [x2, 48]
.L321:
	mov	w0, w12
	cmp	w12, w1
	beq	.L322
.L323:
	add	w3, w9, w0
	add	w10, w4, w0
	add	w8, w0, 1
	sxtw	x3, w3
	ldr	s0, [x20, w10, sxtw 2]
	ldr	s3, [x6, x3, lsl 2]
	fmadd	s0, s3, s1, s0
	str	s0, [x6, x3, lsl 2]
	cmp	w1, w8
	ble	.L322
	add	w3, w9, w8
	add	w8, w4, w8
	add	w0, w0, 2
	sxtw	x3, w3
	ldr	s0, [x20, w8, sxtw 2]
	ldr	s3, [x6, x3, lsl 2]
	fmadd	s0, s3, s1, s0
	str	s0, [x6, x3, lsl 2]
	cmp	w0, w1
	bge	.L322
	add	w3, w9, w0
	add	w0, w4, w0
	sxtw	x3, w3
	ldr	s0, [x20, w0, sxtw 2]
	ldr	s3, [x6, x3, lsl 2]
	fmadd	s0, s3, s1, s0
	str	s0, [x6, x3, lsl 2]
.L322:
	add	w4, w4, 4
	add	w9, w9, w7
	add	x2, x2, x14
	add	x5, x5, 16
	cmp	w19, w4
	bne	.L320
	ldp	x19, x20, [sp, 16]
	.cfi_remember_state
	.cfi_restore 20
	.cfi_restore 19
	b	.L306
.L343:
	.cfi_restore_state
	mov	v7.16b, v6.16b
	mov	v16.16b, v6.16b
	mov	v17.16b, v6.16b
	mov	v18.16b, v6.16b
	mov	v19.16b, v6.16b
	mov	v20.16b, v6.16b
	mov	v21.16b, v6.16b
	mov	v23.16b, v6.16b
	mov	v24.16b, v6.16b
	mov	v25.16b, v6.16b
	mov	v26.16b, v6.16b
	mov	v27.16b, v6.16b
	mov	v28.16b, v6.16b
	mov	v29.16b, v6.16b
	mov	v22.16b, v6.16b
	b	.L310
.L405:
	mov	w0, 0
	b	.L323
.L402:
	.cfi_restore 19
	.cfi_restore 20
	stp	x19, x20, [sp, 16]
	.cfi_offset 20, -120
	.cfi_offset 19, -128
	stp	x21, x22, [sp, 32]
	.cfi_offset 22, -104
	.cfi_offset 21, -112
	str	x23, [sp, 48]
	.cfi_offset 23, -96
	str	d8, [sp, 56]
	.cfi_offset 72, -88
	bl	__stack_chk_fail
.L338:
	.cfi_restore 21
	.cfi_restore 22
	.cfi_restore 23
	.cfi_restore 72
	mov	w2, w1
	mov	w3, w19
	adrp	x1, .LC1
	add	x1, x1, :lo12:.LC1
	mov	w0, 1
	stp	x21, x22, [sp, 32]
	.cfi_offset 22, -104
	.cfi_offset 21, -112
	str	x23, [sp, 48]
	.cfi_offset 23, -96
	str	d8, [sp, 56]
	.cfi_offset 72, -88
	bl	__printf_chk
	mov	w0, -1
	bl	exit
	.cfi_endproc
.LFE4310:
	.size	gemm_microkernel_Cresident_neon_8x8_fp32, .-gemm_microkernel_Cresident_neon_8x8_fp32
	.align	2
	.p2align 4,,11
	.global	gemm_microkernel_Cresident_neon_8x8_prefetch_fp32
	.type	gemm_microkernel_Cresident_neon_8x8_prefetch_fp32, %function
gemm_microkernel_Cresident_neon_8x8_prefetch_fp32:
.LFB4312:
	.cfi_startproc
	stp	x29, x30, [sp, -144]!
	.cfi_def_cfa_offset 144
	.cfi_offset 29, -144
	.cfi_offset 30, -136
	adrp	x8, :got:__stack_chk_guard
	mov	x29, sp
	ldr	x8, [x8, #:got_lo12:__stack_chk_guard]
	ldr	x9, [x8]
	str	x9, [sp, 136]
	mov	x9, 0
	cbz	w3, .L406
	and	w0, w0, 255
	stp	x19, x20, [sp, 16]
	.cfi_offset 20, -120
	.cfi_offset 19, -128
	cmp	w0, 67
	movi	v6.4s, 0
	mov	w20, w2
	csel	x2, x4, x5, eq
	csel	x5, x5, x4, eq
	fcmp	s0, #0.0
	mov	x8, x2
	ldp	q5, q3, [x5]
	ldr	q26, [x8], 16
	ldr	q2, [x2, 16]
	beq	.L442
	cmp	w3, 1
	ble	.L443
	sub	w3, w3, #2
	mov	x9, 32
	mov	v27.16b, v6.16b
	add	x4, x5, x9
	mov	v28.16b, v6.16b
	add	x3, x9, x3, uxtw 4
	mov	v29.16b, v6.16b
	add	x3, x2, x3
	mov	v22.16b, v6.16b
	mov	v23.16b, v6.16b
	mov	v24.16b, v6.16b
	mov	v25.16b, v6.16b
	mov	v30.16b, v6.16b
	mov	v31.16b, v6.16b
	mov	v16.16b, v6.16b
	mov	v17.16b, v6.16b
	mov	v18.16b, v6.16b
	mov	v19.16b, v6.16b
	mov	v20.16b, v6.16b
	mov	v21.16b, v6.16b
	mov	v7.16b, v2.16b
	.p2align 3,,7
.L411:
	mov	v4.16b, v26.16b
	mov	v26.16b, v7.16b
	ldr	q7, [x8, 16]!
	fmla	v25.4s, v2.4s, v5.s[0]
	fmla	v24.4s, v2.4s, v5.s[1]
	fmla	v23.4s, v2.4s, v5.s[2]
	fmla	v22.4s, v2.4s, v5.s[3]
	fmla	v29.4s, v2.4s, v3.s[0]
	fmla	v28.4s, v2.4s, v3.s[1]
	fmla	v27.4s, v2.4s, v3.s[2]
	fmla	v6.4s, v2.4s, v3.s[3]
	fmla	v21.4s, v4.4s, v5.s[0]
	fmla	v20.4s, v4.4s, v5.s[1]
	fmla	v19.4s, v4.4s, v5.s[2]
	fmla	v18.4s, v4.4s, v5.s[3]
	fmla	v17.4s, v4.4s, v3.s[0]
	fmla	v16.4s, v4.4s, v3.s[1]
	fmla	v31.4s, v4.4s, v3.s[2]
	fmla	v30.4s, v4.4s, v3.s[3]
	mov	v2.16b, v7.16b
	ldr	q5, [x4, -16]
	ldr	q3, [x4], 16
	cmp	x8, x3
	bne	.L411
.L410:
	fmla	v21.4s, v26.4s, v5.s[0]
	fmov	s4, -1.0e+0
	fmla	v20.4s, v26.4s, v5.s[1]
	fmla	v19.4s, v26.4s, v5.s[2]
	fcmp	s0, s4
	fmla	v18.4s, v26.4s, v5.s[3]
	fmla	v25.4s, v2.4s, v5.s[0]
	fmla	v24.4s, v2.4s, v5.s[1]
	fmla	v23.4s, v2.4s, v5.s[2]
	fmla	v22.4s, v2.4s, v5.s[3]
	mov	v7.16b, v29.16b
	mov	v5.16b, v28.16b
	mov	v4.16b, v27.16b
	fmla	v17.4s, v26.4s, v3.s[0]
	fmla	v16.4s, v26.4s, v3.s[1]
	fmla	v31.4s, v26.4s, v3.s[2]
	fmla	v30.4s, v26.4s, v3.s[3]
	fmla	v7.4s, v2.4s, v3.s[0]
	fmla	v5.4s, v2.4s, v3.s[1]
	fmla	v4.4s, v2.4s, v3.s[2]
	fmla	v6.4s, v2.4s, v3.s[3]
	beq	.L501
	fmov	s2, 1.0e+0
	fcmp	s0, s2
	beq	.L409
	dup	v0.4s, v0.s[0]
	fmul	v21.4s, v21.4s, v0.4s
	fmul	v20.4s, v20.4s, v0.4s
	fmul	v19.4s, v19.4s, v0.4s
	fmul	v18.4s, v18.4s, v0.4s
	fmul	v17.4s, v17.4s, v0.4s
	fmul	v16.4s, v16.4s, v0.4s
	fmul	v31.4s, v31.4s, v0.4s
	fmul	v30.4s, v30.4s, v0.4s
	fmul	v25.4s, v25.4s, v0.4s
	fmul	v24.4s, v24.4s, v0.4s
	fmul	v23.4s, v23.4s, v0.4s
	fmul	v22.4s, v22.4s, v0.4s
	fmul	v7.4s, v7.4s, v0.4s
	fmul	v5.4s, v5.4s, v0.4s
	fmul	v4.4s, v4.4s, v0.4s
	fmul	v6.4s, v6.4s, v0.4s
	b	.L409
	.p2align 2,,3
.L413:
	cmp	w1, 4
	ccmp	w20, 4, 0, eq
	bne	.L438
	lsl	w8, w7, 1
	sbfiz	x1, x7, 2, 32
	add	w0, w8, w7
	lsl	w2, w7, 3
	sub	w2, w2, w7
	add	w5, w7, w7, lsl 2
	lsl	w4, w0, 1
	add	x12, x6, x1
	add	x11, x12, x1
	fcmp	s1, #0.0
	sxtw	x3, w0
	add	x10, x11, x1
	sxtw	x0, w7
	sxtw	x4, w4
	sxtw	x2, w2
	sxtw	x5, w5
	add	x9, x10, x1
	add	x4, x4, 4
	add	x3, x3, 4
	add	x2, x2, 4
	add	x5, x5, 4
	add	x0, x0, 1
	mov	x13, 16
	add	x7, x9, x1
	add	x8, x13, x8, sxtw 2
	add	x14, x7, x1
	lsl	x4, x4, 2
	lsl	x3, x3, 2
	lsl	x2, x2, 2
	lsl	x5, x5, 2
	lsl	x0, x0, 4
	add	x13, x1, x13
	beq	.L439
	str	d8, [sp, 56]
	.cfi_offset 72, -88
	ldr	q8, [x9, x1]
	ldr	q26, [x6, x1]
	fmla	v16.4s, v8.4s, v1.s[0]
	ldr	q8, [x7, x1]
	ldr	q3, [x12, x1]
	fmla	v31.4s, v8.4s, v1.s[0]
	ldr	q8, [x14, x1]
	ldr	q2, [x11, x1]
	fmla	v30.4s, v8.4s, v1.s[0]
	ldr	q0, [x10, x1]
	ldp	q27, q8, [x6]
	fmla	v20.4s, v26.4s, v1.s[0]
	fmla	v19.4s, v3.4s, v1.s[0]
	fmla	v18.4s, v2.4s, v1.s[0]
	fmla	v17.4s, v0.4s, v1.s[0]
	fmla	v21.4s, v27.4s, v1.s[0]
	ldr	q29, [x6, x13]
	ldr	q28, [x6, x8]
	ldr	q26, [x6, x0]
	ldr	q3, [x6, x5]
	ldr	q2, [x6, x4]
	ldr	q0, [x6, x2]
	ldr	q27, [x6, x3]
	fmla	v24.4s, v29.4s, v1.s[0]
	fmla	v23.4s, v28.4s, v1.s[0]
	fmla	v7.4s, v26.4s, v1.s[0]
	fmla	v5.4s, v3.4s, v1.s[0]
	fmla	v4.4s, v2.4s, v1.s[0]
	fmla	v6.4s, v0.4s, v1.s[0]
	fmla	v22.4s, v27.4s, v1.s[0]
	fmla	v25.4s, v8.4s, v1.s[0]
	ldr	d8, [sp, 56]
	.cfi_restore 72
.L439:
	str	q21, [x6]
	str	q20, [x6, x1]
	str	q19, [x12, x1]
	str	q18, [x11, x1]
	str	q17, [x10, x1]
	str	q16, [x9, x1]
	str	q31, [x7, x1]
	str	q30, [x14, x1]
	str	q25, [x6, 16]
	str	q24, [x6, x13]
	str	q23, [x6, x8]
	str	q22, [x6, x3]
	str	q7, [x6, x0]
	str	q5, [x6, x5]
	str	q4, [x6, x4]
	str	q6, [x6, x2]
	ldp	x19, x20, [sp, 16]
	.cfi_restore 20
	.cfi_restore 19
.L406:
	adrp	x0, :got:__stack_chk_guard
	ldr	x0, [x0, #:got_lo12:__stack_chk_guard]
	ldr	x1, [sp, 136]
	ldr	x2, [x0]
	subs	x1, x1, x2
	mov	x2, 0
	bne	.L502
	ldp	x29, x30, [sp], 144
	.cfi_restore 30
	.cfi_restore 29
	.cfi_def_cfa_offset 0
	ret
	.p2align 2,,3
.L442:
	.cfi_def_cfa_offset 144
	.cfi_offset 19, -128
	.cfi_offset 20, -120
	.cfi_offset 29, -144
	.cfi_offset 30, -136
	mov	v4.16b, v6.16b
	mov	v5.16b, v6.16b
	mov	v7.16b, v6.16b
	mov	v22.16b, v6.16b
	mov	v23.16b, v6.16b
	mov	v24.16b, v6.16b
	mov	v25.16b, v6.16b
	mov	v30.16b, v6.16b
	mov	v31.16b, v6.16b
	mov	v16.16b, v6.16b
	mov	v17.16b, v6.16b
	mov	v18.16b, v6.16b
	mov	v19.16b, v6.16b
	mov	v20.16b, v6.16b
	mov	v21.16b, v6.16b
.L409:
	cmp	w1, 3
	ccmp	w20, 3, 4, gt
	bgt	.L413
	fcmp	s1, #0.0
	stp	q21, q25, [sp, 64]
	add	x19, sp, 64
	stp	q24, q23, [sp, 96]
	stp	q22, q7, [sp, 128]
	stp	q5, q4, [sp, 160]
	str	q6, [sp, 192]
	beq	.L414
	cmp	w0, 67
	beq	.L415
	cmp	w20, 0
	ble	.L498
	cmp	w1, 0
	ble	.L498
	dup	v2.4s, v1.s[0]
	sub	w11, w20, #1
	lsr	w9, w20, 2
	and	w10, w20, -4
	and	w12, w20, 3
	mov	w8, 0
	mov	w5, 0
	.p2align 3,,7
.L418:
	cmp	w11, 2
	bls	.L503
	sbfiz	x3, x5, 2, 32
	sbfiz	x0, x8, 2, 32
	add	x2, x6, x3
	add	x4, x19, x0
	ldr	q3, [x6, x3]
	ldr	q0, [x19, x0]
	fmla	v0.4s, v2.4s, v3.4s
	str	q0, [x6, x3]
	cmp	w9, 1
	bls	.L425
	ldr	q3, [x2, 16]
	ldr	q0, [x4, 16]
	fmla	v0.4s, v3.4s, v2.4s
	str	q0, [x2, 16]
	cmp	w9, 2
	beq	.L425
	ldr	q3, [x2, 32]
	ldr	q0, [x4, 32]
	fmla	v0.4s, v3.4s, v2.4s
	str	q0, [x2, 32]
	cmp	w9, 3
	beq	.L425
	ldr	q3, [x2, 48]
	ldr	q0, [x4, 48]
	fmla	v0.4s, v3.4s, v2.4s
	str	q0, [x2, 48]
.L425:
	cmp	w20, w10
	beq	.L426
	mov	w13, w12
	mov	w0, w10
.L427:
	add	w2, w0, w5
	add	w4, w0, w8
	add	w3, w0, 1
	sxtw	x2, w2
	ldr	s0, [x19, w4, sxtw 2]
	ldr	s3, [x6, x2, lsl 2]
	fmadd	s0, s3, s1, s0
	str	s0, [x6, x2, lsl 2]
	cmp	w13, 1
	beq	.L426
	add	w2, w5, w3
	add	w3, w8, w3
	add	w0, w0, 2
	sxtw	x2, w2
	ldr	s0, [x19, w3, sxtw 2]
	ldr	s3, [x6, x2, lsl 2]
	fmadd	s0, s1, s3, s0
	str	s0, [x6, x2, lsl 2]
	cmp	w13, 2
	beq	.L426
	add	w2, w5, w0
	add	w0, w8, w0
	sxtw	x2, w2
	ldr	s0, [x19, w0, sxtw 2]
	ldr	s3, [x6, x2, lsl 2]
	fmadd	s0, s1, s3, s0
	str	s0, [x6, x2, lsl 2]
.L426:
	add	w5, w5, w7
	add	w8, w8, 4
	subs	w1, w1, #1
	bne	.L418
.L498:
	ldp	x19, x20, [sp, 16]
	.cfi_remember_state
	.cfi_restore 20
	.cfi_restore 19
	b	.L406
	.p2align 2,,3
.L501:
	.cfi_restore_state
	fneg	v21.4s, v21.4s
	fneg	v20.4s, v20.4s
	fneg	v19.4s, v19.4s
	fneg	v18.4s, v18.4s
	fneg	v17.4s, v17.4s
	fneg	v16.4s, v16.4s
	fneg	v31.4s, v31.4s
	fneg	v30.4s, v30.4s
	fneg	v25.4s, v25.4s
	fneg	v24.4s, v24.4s
	fneg	v23.4s, v23.4s
	fneg	v22.4s, v22.4s
	fneg	v7.4s, v7.4s
	fneg	v5.4s, v5.4s
	fneg	v4.4s, v4.4s
	fneg	v6.4s, v6.4s
	b	.L409
	.p2align 2,,3
.L414:
	cmp	w0, 67
	beq	.L429
	cmp	w20, 0
	ble	.L498
	cmp	w1, 0
	ble	.L498
	sub	w12, w20, #1
	lsr	w9, w20, 2
	and	w10, w20, -4
	and	w13, w20, 3
	mov	w3, 0
	mov	w2, 0
	.p2align 3,,7
.L431:
	cmp	w12, 2
	bls	.L504
	sbfiz	x0, x2, 2, 32
	sbfiz	x4, x3, 2, 32
	add	x8, x19, x0
	add	x5, x6, x4
	ldr	q0, [x19, x0]
	str	q0, [x6, x4]
	cmp	w9, 1
	bls	.L434
	ldr	q0, [x8, 16]
	str	q0, [x5, 16]
	cmp	w9, 2
	beq	.L434
	ldr	q0, [x8, 32]
	str	q0, [x5, 32]
	cmp	w9, 3
	beq	.L434
	ldr	q0, [x8, 48]
	str	q0, [x5, 48]
.L434:
	cmp	w20, w10
	beq	.L435
	mov	w8, w13
	mov	w0, w10
.L436:
	add	w5, w0, w2
	add	w11, w0, w3
	add	w4, w0, 1
	ldr	s0, [x19, w5, sxtw 2]
	str	s0, [x6, w11, sxtw 2]
	cmp	w8, 1
	beq	.L435
	add	w5, w2, w4
	add	w4, w3, w4
	add	w0, w0, 2
	ldr	s0, [x19, w5, sxtw 2]
	str	s0, [x6, w4, sxtw 2]
	cmp	w8, 2
	beq	.L435
	add	w4, w2, w0
	add	w0, w3, w0
	ldr	s0, [x19, w4, sxtw 2]
	str	s0, [x6, w0, sxtw 2]
.L435:
	add	w2, w2, 4
	add	w3, w3, w7
	subs	w1, w1, #1
	bne	.L431
	ldp	x19, x20, [sp, 16]
	.cfi_remember_state
	.cfi_restore 20
	.cfi_restore 19
	b	.L406
	.p2align 2,,3
.L504:
	.cfi_restore_state
	mov	w8, w20
	mov	w0, 0
	b	.L436
	.p2align 2,,3
.L503:
	mov	w13, w20
	mov	w0, 0
	b	.L427
	.p2align 2,,3
.L429:
	cmp	w20, 0
	ble	.L498
	cmp	w1, 0
	ble	.L498
	stp	x21, x22, [sp, 32]
	.cfi_offset 22, -104
	.cfi_offset 21, -112
	sbfiz	x22, x1, 2, 32
	mov	w21, 0
	str	x23, [sp, 48]
	.cfi_offset 23, -96
	sbfiz	x23, x7, 2, 32
.L433:
	mov	x1, x19
	mov	x0, x6
	mov	x2, x22
	add	w21, w21, 1
	bl	memcpy
	add	x19, x19, 16
	add	x6, x0, x23
	cmp	w20, w21
	bne	.L433
	ldp	x19, x20, [sp, 16]
	.cfi_restore 20
	.cfi_restore 19
	ldp	x21, x22, [sp, 32]
	.cfi_restore 22
	.cfi_restore 21
	ldr	x23, [sp, 48]
	.cfi_restore 23
	b	.L406
	.p2align 2,,3
.L415:
	.cfi_offset 19, -128
	.cfi_offset 20, -120
	cmp	w20, 0
	ble	.L498
	cmp	w1, 0
	ble	.L498
	dup	v2.4s, v1.s[0]
	lsl	w20, w20, 2
	sbfiz	x14, x7, 2, 32
	mov	x2, x6
	mov	x5, x19
	sub	w13, w1, #1
	lsr	w11, w1, 2
	and	w12, w1, -4
	mov	w4, 0
	mov	w9, 0
.L420:
	cmp	w13, 2
	bls	.L505
	ldr	q3, [x2]
	ldr	q0, [x5]
	fmla	v0.4s, v3.4s, v2.4s
	str	q0, [x2]
	cmp	w11, 1
	bls	.L421
	ldr	q3, [x2, 16]
	ldr	q0, [x5, 16]
	fmla	v0.4s, v2.4s, v3.4s
	str	q0, [x2, 16]
	cmp	w11, 2
	beq	.L421
	ldr	q3, [x2, 32]
	ldr	q0, [x5, 32]
	fmla	v0.4s, v2.4s, v3.4s
	str	q0, [x2, 32]
	cmp	w11, 3
	beq	.L421
	ldr	q3, [x2, 48]
	ldr	q0, [x5, 48]
	fmla	v0.4s, v2.4s, v3.4s
	str	q0, [x2, 48]
.L421:
	mov	w0, w12
	cmp	w12, w1
	beq	.L422
.L423:
	add	w3, w9, w0
	add	w10, w4, w0
	add	w8, w0, 1
	sxtw	x3, w3
	ldr	s0, [x19, w10, sxtw 2]
	ldr	s3, [x6, x3, lsl 2]
	fmadd	s0, s3, s1, s0
	str	s0, [x6, x3, lsl 2]
	cmp	w1, w8
	ble	.L422
	add	w3, w9, w8
	add	w8, w4, w8
	add	w0, w0, 2
	sxtw	x3, w3
	ldr	s0, [x19, w8, sxtw 2]
	ldr	s3, [x6, x3, lsl 2]
	fmadd	s0, s3, s1, s0
	str	s0, [x6, x3, lsl 2]
	cmp	w0, w1
	bge	.L422
	add	w3, w9, w0
	add	w0, w4, w0
	sxtw	x3, w3
	ldr	s0, [x19, w0, sxtw 2]
	ldr	s3, [x6, x3, lsl 2]
	fmadd	s0, s3, s1, s0
	str	s0, [x6, x3, lsl 2]
.L422:
	add	w4, w4, 4
	add	w9, w9, w7
	add	x2, x2, x14
	add	x5, x5, 16
	cmp	w20, w4
	bne	.L420
	ldp	x19, x20, [sp, 16]
	.cfi_remember_state
	.cfi_restore 20
	.cfi_restore 19
	b	.L406
.L443:
	.cfi_restore_state
	mov	v27.16b, v6.16b
	mov	v28.16b, v6.16b
	mov	v29.16b, v6.16b
	mov	v22.16b, v6.16b
	mov	v23.16b, v6.16b
	mov	v24.16b, v6.16b
	mov	v25.16b, v6.16b
	mov	v30.16b, v6.16b
	mov	v31.16b, v6.16b
	mov	v16.16b, v6.16b
	mov	v17.16b, v6.16b
	mov	v18.16b, v6.16b
	mov	v19.16b, v6.16b
	mov	v20.16b, v6.16b
	mov	v21.16b, v6.16b
	b	.L410
.L505:
	mov	w0, 0
	b	.L423
.L502:
	.cfi_restore 19
	.cfi_restore 20
	stp	x19, x20, [sp, 16]
	.cfi_offset 20, -120
	.cfi_offset 19, -128
	stp	x21, x22, [sp, 32]
	.cfi_offset 22, -104
	.cfi_offset 21, -112
	str	x23, [sp, 48]
	.cfi_offset 23, -96
	str	d8, [sp, 56]
	.cfi_offset 72, -88
	bl	__stack_chk_fail
.L438:
	.cfi_restore 21
	.cfi_restore 22
	.cfi_restore 23
	.cfi_restore 72
	mov	w2, w1
	mov	w3, w20
	adrp	x1, .LC0
	add	x1, x1, :lo12:.LC0
	mov	w0, 1
	stp	x21, x22, [sp, 32]
	.cfi_offset 22, -104
	.cfi_offset 21, -112
	str	x23, [sp, 48]
	.cfi_offset 23, -96
	str	d8, [sp, 56]
	.cfi_offset 72, -88
	bl	__printf_chk
	mov	w0, -1
	bl	exit
	.cfi_endproc
.LFE4312:
	.size	gemm_microkernel_Cresident_neon_8x8_prefetch_fp32, .-gemm_microkernel_Cresident_neon_8x8_prefetch_fp32
	.align	2
	.p2align 4,,11
	.global	gemm_microkernel_ABresident_neon_4x4_fp32
	.type	gemm_microkernel_ABresident_neon_4x4_fp32, %function
gemm_microkernel_ABresident_neon_4x4_fp32:
.LFB4313:
	.cfi_startproc
	stp	x29, x30, [sp, -256]!
	.cfi_def_cfa_offset 256
	.cfi_offset 29, -256
	.cfi_offset 30, -248
	adrp	x8, :got:__stack_chk_guard
	mov	x29, sp
	ldr	x8, [x8, #:got_lo12:__stack_chk_guard]
	stp	x19, x20, [sp, 16]
	ldr	x9, [x8]
	str	x9, [sp, 248]
	mov	x9, 0
	.cfi_offset 19, -240
	.cfi_offset 20, -232
	ldr	x19, [sp, 256]
	cbz	w3, .L506
	fcmp	s0, #0.0
	stp	x21, x22, [sp, 32]
	.cfi_offset 22, -216
	.cfi_offset 21, -224
	mov	w20, w2
	stp	d8, d9, [sp, 96]
	.cfi_offset 73, -152
	.cfi_offset 72, -160
	mov	w21, w3
	fmov	s9, s0
	fmov	s8, s1
	bne	.L508
	cmp	w2, 4
	beq	.L509
	cmp	w3, 0
	ble	.L685
	lsr	w6, w2, 2
	mov	x4, x19
	dup	v1.4s, v1.s[0]
	sub	w7, w2, #1
	lsl	x6, x6, 4
	and	w5, w2, -4
	lsl	w21, w3, 2
	fmov	s2, -1.0e+0
	mov	w3, 0
	fmov	s3, 1.0e+0
	.p2align 3,,7
.L520:
	cmp	w20, 0
	ble	.L530
	fcmp	s8, s2
	beq	.L521
	fcmp	s8, s3
	beq	.L530
	cmp	w7, 2
	bls	.L577
	add	x1, x6, x4
	mov	x0, x4
	.p2align 3,,7
.L524:
	ldr	q0, [x0]
	fmul	v0.4s, v0.4s, v1.4s
	str	q0, [x0], 16
	cmp	x0, x1
	bne	.L524
	mov	w0, w5
	cmp	w20, w5
	beq	.L530
.L523:
	add	w2, w0, w3
	add	w1, w0, 1
	sxtw	x2, w2
	ldr	s0, [x19, x2, lsl 2]
	fmul	s0, s0, s8
	str	s0, [x19, x2, lsl 2]
	cmp	w20, w1
	ble	.L530
	add	w1, w3, w1
	add	w0, w0, 2
	sxtw	x1, w1
	ldr	s0, [x19, x1, lsl 2]
	fmul	s0, s0, s8
	str	s0, [x19, x1, lsl 2]
	cmp	w20, w0
	ble	.L530
	add	w0, w3, w0
	sxtw	x0, w0
	ldr	s0, [x19, x0, lsl 2]
	fmul	s0, s0, s8
	str	s0, [x19, x0, lsl 2]
	.p2align 3,,7
.L530:
	add	w3, w3, 4
	add	x4, x4, 16
	cmp	w3, w21
	bne	.L520
.L685:
	ldp	x21, x22, [sp, 32]
	.cfi_restore 22
	.cfi_restore 21
	ldp	d8, d9, [sp, 96]
	.cfi_restore 73
	.cfi_restore 72
.L506:
	adrp	x0, :got:__stack_chk_guard
	ldr	x0, [x0, #:got_lo12:__stack_chk_guard]
	ldr	x1, [sp, 248]
	ldr	x2, [x0]
	subs	x1, x1, x2
	mov	x2, 0
	bne	.L711
	ldp	x19, x20, [sp, 16]
	ldp	x29, x30, [sp], 256
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 19
	.cfi_restore 20
	.cfi_def_cfa_offset 0
	ret
	.p2align 2,,3
.L508:
	.cfi_def_cfa_offset 256
	.cfi_offset 19, -240
	.cfi_offset 20, -232
	.cfi_offset 21, -224
	.cfi_offset 22, -216
	.cfi_offset 29, -256
	.cfi_offset 30, -248
	.cfi_offset 72, -160
	.cfi_offset 73, -152
	stp	x23, x24, [sp, 48]
	.cfi_offset 24, -200
	.cfi_offset 23, -208
	cmp	w2, 3
	mov	x23, x5
	stp	x27, x28, [sp, 80]
	.cfi_offset 28, -168
	.cfi_offset 27, -176
	mov	w24, w6
	mov	w28, w4
	mov	x22, x7
	and	w0, w0, 255
	and	w1, w1, 255
	ccmp	w4, 3, 4, gt
	bgt	.L532
	cmp	w1, 78
	beq	.L712
	cmp	w0, 67
	beq	.L552
	cmp	w4, 0
	ble	.L708
	sub	w27, w2, #1
	stp	x25, x26, [sp, 64]
	.cfi_offset 26, -184
	.cfi_offset 25, -192
	add	x0, x27, 1
	mov	w26, 3
	sub	w26, w26, w2
	add	x26, x26, 1
	lsl	x0, x0, 2
	add	x27, sp, 176
	str	x0, [sp, 128]
	sxtw	x0, w2
	mov	w25, 0
	str	x0, [sp, 144]
	lsl	x0, x26, 2
	mov	w26, 0
	str	x0, [sp, 136]
	.p2align 3,,7
.L564:
	lsl	w5, w26, 2
	cmp	w20, 0
	ble	.L563
	ldr	x2, [sp, 128]
	add	x0, x27, x5, sxtw 2
	add	x1, x23, x25, sxtw 2
	str	w5, [sp, 120]
	bl	memcpy
	ldr	w5, [sp, 120]
	cmp	w20, 3
	bgt	.L566
.L563:
	ldp	x2, x0, [sp, 136]
	mov	w1, 0
	add	x0, x0, x5, sxtw
	add	x0, x27, x0, lsl 2
	bl	memset
.L566:
	add	w26, w26, 1
	add	w25, w25, w24
	cmp	w28, w26
	bne	.L564
.L683:
	ldp	x25, x26, [sp, 64]
	.cfi_restore 26
	.cfi_restore 25
	cmp	w28, 3
	bgt	.L545
.L554:
	lsl	w1, w28, 2
	mov	w0, 4
	movi	v0.4s, 0
	sub	w4, w0, w28
	sbfiz	x0, x1, 2, 32
	add	x1, x27, x0
	str	q0, [x27, x0]
	cmp	w4, 1
	bls	.L545
	str	q0, [x1, 16]
	cmp	w4, 2
	beq	.L545
	str	q0, [x1, 32]
	cmp	w4, 3
	beq	.L545
	str	q0, [x1, 48]
.L545:
	ldp	q5, q3, [sp, 176]
	ldp	q4, q2, [sp, 208]
.L544:
	fmov	s0, -1.0e+0
	fcmp	s9, s0
	bne	.L570
.L713:
	fneg	v5.4s, v5.4s
	fneg	v3.4s, v3.4s
	fneg	v4.4s, v4.4s
	fneg	v2.4s, v2.4s
.L571:
	fcmp	s8, #0.0
	beq	.L572
	cmp	w21, 0
	ble	.L690
	sub	w0, w21, #1
	add	x2, x19, 16
	dup	v8.4s, v8.s[0]
	mov	x1, x22
	add	x0, x2, x0, uxtw 4
	.p2align 3,,7
.L575:
	ldr	q0, [x19]
	ldr	q1, [x1], 16
	fmul	v0.4s, v0.4s, v8.4s
	fmla	v0.4s, v5.4s, v1.s[0]
	fmla	v0.4s, v3.4s, v1.s[1]
	fmla	v0.4s, v4.4s, v1.s[2]
	fmla	v0.4s, v2.4s, v1.s[3]
	str	q0, [x19], 16
	cmp	x19, x0
	bne	.L575
.L690:
	ldp	x21, x22, [sp, 32]
	.cfi_remember_state
	.cfi_restore 22
	.cfi_restore 21
	ldp	x23, x24, [sp, 48]
	.cfi_restore 24
	.cfi_restore 23
	ldp	x27, x28, [sp, 80]
	.cfi_restore 28
	.cfi_restore 27
	ldp	d8, d9, [sp, 96]
	.cfi_restore 73
	.cfi_restore 72
	b	.L506
	.p2align 2,,3
.L532:
	.cfi_restore_state
	cmp	w2, 4
	ccmp	w4, 4, 0, eq
	bne	.L568
	sbfiz	x2, x6, 2, 32
	cmp	w1, 84
	add	x3, x5, x2
	mov	w5, 67
	add	x4, x3, x2
	ccmp	w0, w5, 0, eq
	ldr	q3, [x23, x2]
	ldr	q2, [x4, x2]
	ldr	q4, [x3, x2]
	ldr	q5, [x23]
	beq	.L569
	cmp	w1, 78
	mov	w1, 82
	ccmp	w0, w1, 0, eq
	bne	.L544
.L569:
	mov	v6.16b, v3.16b
	zip2	v0.2d, v5.2d, v4.2d
	zip1	v3.2d, v3.2d, v2.2d
	zip1	v1.2d, v5.2d, v4.2d
	zip2	v2.2d, v6.2d, v2.2d
	trn1	v5.4s, v1.4s, v3.4s
	trn1	v4.4s, v0.4s, v2.4s
	trn2	v2.4s, v0.4s, v2.4s
	fmov	s0, -1.0e+0
	trn2	v3.4s, v1.4s, v3.4s
	fcmp	s9, s0
	beq	.L713
.L570:
	fmov	s0, 1.0e+0
	fcmp	s9, s0
	beq	.L571
	dup	v9.4s, v9.s[0]
	fmul	v5.4s, v5.4s, v9.4s
	fmul	v3.4s, v3.4s, v9.4s
	fmul	v4.4s, v4.4s, v9.4s
	fmul	v2.4s, v2.4s, v9.4s
	b	.L571
	.p2align 2,,3
.L521:
	.cfi_restore 23
	.cfi_restore 24
	.cfi_restore 27
	.cfi_restore 28
	cmp	w7, 2
	bls	.L578
	add	x1, x6, x4
	mov	x0, x4
	.p2align 3,,7
.L528:
	ldr	q0, [x0]
	fneg	v0.4s, v0.4s
	str	q0, [x0], 16
	cmp	x0, x1
	bne	.L528
	mov	w0, w5
	cmp	w20, w5
	beq	.L530
.L527:
	add	w2, w3, w0
	add	w1, w0, 1
	sxtw	x2, w2
	ldr	s0, [x19, x2, lsl 2]
	fneg	s0, s0
	str	s0, [x19, x2, lsl 2]
	cmp	w20, w1
	ble	.L530
	add	w1, w3, w1
	add	w0, w0, 2
	sxtw	x1, w1
	ldr	s0, [x19, x1, lsl 2]
	fneg	s0, s0
	str	s0, [x19, x1, lsl 2]
	cmp	w20, w0
	ble	.L530
	add	w0, w3, w0
	sxtw	x0, w0
	ldr	s0, [x19, x0, lsl 2]
	fneg	s0, s0
	str	s0, [x19, x0, lsl 2]
	b	.L530
	.p2align 2,,3
.L572:
	.cfi_offset 23, -208
	.cfi_offset 24, -200
	.cfi_offset 27, -176
	.cfi_offset 28, -168
	cmp	w21, 0
	ble	.L690
	sub	w1, w21, #1
	add	x2, x22, 16
	movi	v6.4s, 0
	mov	x0, x22
	add	x1, x2, x1, uxtw 4
	.p2align 3,,7
.L574:
	mov	v0.16b, v6.16b
	ldr	q1, [x0], 16
	fmla	v0.4s, v5.4s, v1.s[0]
	fmla	v0.4s, v3.4s, v1.s[1]
	fmla	v0.4s, v4.4s, v1.s[2]
	fmla	v0.4s, v2.4s, v1.s[3]
	str	q0, [x19], 16
	cmp	x0, x1
	bne	.L574
	ldp	x21, x22, [sp, 32]
	.cfi_restore 22
	.cfi_restore 21
	ldp	x23, x24, [sp, 48]
	.cfi_restore 24
	.cfi_restore 23
	ldp	x27, x28, [sp, 80]
	.cfi_restore 28
	.cfi_restore 27
	ldp	d8, d9, [sp, 96]
	.cfi_restore 73
	.cfi_restore 72
	b	.L506
.L509:
	.cfi_offset 21, -224
	.cfi_offset 22, -216
	.cfi_offset 72, -160
	.cfi_offset 73, -152
	cmp	w3, 0
	ble	.L685
	fmov	s0, -1.0e+0
	fcmp	s1, s0
	beq	.L513
	fmov	s0, 1.0e+0
	sub	w0, w3, #1
	fcmp	s1, s0
	bne	.L514
	add	x1, x19, 16
	add	x0, x1, x0, uxtw 4
	.p2align 3,,7
.L515:
	add	x19, x19, 16
	cmp	x19, x0
	bne	.L515
	b	.L685
.L578:
	mov	w0, 0
	b	.L527
.L712:
	.cfi_offset 23, -208
	.cfi_offset 24, -200
	.cfi_offset 27, -176
	.cfi_offset 28, -168
	cmp	w0, 67
	beq	.L534
	cmp	w4, 0
	ble	.L708
	lsl	w10, w6, 1
	mov	w2, 3
	add	w0, w10, w6
	sub	w2, w2, w20
	add	x2, x2, 1
	str	w0, [sp, 172]
	add	w0, w4, w6
	lsl	w8, w6, 3
	str	w0, [sp, 156]
	lsl	w0, w6, 2
	sub	w9, w8, w6
	lsl	x2, x2, 2
	add	x27, sp, 176
	stp	x25, x26, [sp, 64]
	.cfi_offset 26, -184
	.cfi_offset 25, -192
	mov	w25, w6
	str	w0, [sp, 168]
	sxtw	x0, w20
	mov	x26, 0
	str	x0, [sp, 160]
	mov	w0, 0
.L548:
	sxtw	x1, w0
	cmp	w20, 0
	ble	.L694
	ldr	s0, [x23, x26, lsl 2]
	str	s0, [x27, x1, lsl 2]
	cmp	w20, 1
	beq	.L694
	add	w7, w0, 1
	ldr	s0, [x23, w25, sxtw 2]
	str	s0, [x27, w7, sxtw 2]
	cmp	w20, 2
	beq	.L694
	add	w7, w10, w26
	add	w12, w0, 2
	ldr	s0, [x23, w7, sxtw 2]
	str	s0, [x27, w12, sxtw 2]
	cmp	w20, 3
	beq	.L694
	add	w7, w10, w25
	add	w12, w0, 3
	ldr	s0, [x23, w7, sxtw 2]
	str	s0, [x27, w12, sxtw 2]
	cmp	w20, 4
	beq	.L714
	ldr	w3, [sp, 168]
	add	w11, w0, 4
	add	w12, w3, w26
	ldr	s0, [x23, w12, sxtw 2]
	str	s0, [x27, w11, sxtw 2]
	cmp	w20, 5
	beq	.L550
	add	w7, w10, w7
	add	w12, w0, 5
	ldr	s0, [x23, w7, sxtw 2]
	str	s0, [x27, w12, sxtw 2]
	cmp	w20, 6
	beq	.L550
	add	w7, w24, w7
	add	w12, w0, 6
	ldr	s0, [x23, w7, sxtw 2]
	str	s0, [x27, w12, sxtw 2]
	cmp	w20, 7
	beq	.L550
	add	w7, w0, 7
	ldr	s0, [x23, w9, sxtw 2]
	str	s0, [x27, w7, sxtw 2]
	cmp	w20, 8
	beq	.L550
	add	w7, w0, 8
	ldr	s0, [x23, w8, sxtw 2]
	str	s0, [x27, w7, sxtw 2]
	cmp	w20, 9
	beq	.L550
	add	w7, w24, w8
	add	w12, w0, 9
	ldr	s0, [x23, w7, sxtw 2]
	str	s0, [x27, w12, sxtw 2]
	cmp	w20, 10
	beq	.L550
	add	w7, w24, w7
	add	w12, w0, 10
	ldr	s0, [x23, w7, sxtw 2]
	str	s0, [x27, w12, sxtw 2]
	cmp	w20, 11
	beq	.L550
	add	w13, w3, w9
	add	w12, w0, 11
	ldr	s0, [x23, w13, sxtw 2]
	str	s0, [x27, w12, sxtw 2]
	cmp	w20, 12
	beq	.L550
	add	w7, w10, w7
	add	w12, w0, 12
	ldr	s0, [x23, w7, sxtw 2]
	str	s0, [x27, w12, sxtw 2]
	cmp	w20, 13
	beq	.L550
	add	w7, w24, w7
	add	w12, w0, 13
	ldr	s0, [x23, w7, sxtw 2]
	str	s0, [x27, w12, sxtw 2]
	cmp	w20, 14
	beq	.L550
	ldr	w3, [sp, 172]
	add	w12, w0, 14
	add	w13, w3, w13
	ldr	s0, [x23, w13, sxtw 2]
	str	s0, [x27, w12, sxtw 2]
	cmp	w20, 15
	beq	.L550
	add	w7, w10, w7
	add	w0, w0, 15
	ldr	s0, [x23, w7, sxtw 2]
	str	s0, [x27, w0, sxtw 2]
	cmp	w20, 3
	bgt	.L550
.L547:
	ldr	x0, [sp, 160]
	str	x2, [sp, 120]
	str	w10, [sp, 128]
	add	x0, x1, x0
	mov	w1, 0
	str	w8, [sp, 136]
	add	x0, x27, x0, lsl 2
	str	w9, [sp, 144]
	str	w11, [sp, 152]
	bl	memset
	ldr	w10, [sp, 128]
	ldr	w8, [sp, 136]
	ldr	w9, [sp, 144]
	ldr	w11, [sp, 152]
	ldr	x2, [sp, 120]
.L550:
	add	w25, w25, 1
	ldr	w1, [sp, 156]
	add	x26, x26, 1
	mov	w0, w11
	add	w9, w9, 1
	add	w8, w8, 1
	cmp	w25, w1
	bne	.L548
	b	.L683
.L513:
	.cfi_restore 23
	.cfi_restore 24
	.cfi_restore 25
	.cfi_restore 26
	.cfi_restore 27
	.cfi_restore 28
	sub	w0, w3, #1
	add	x1, x19, 16
	add	x0, x1, x0, uxtw 4
	.p2align 3,,7
.L518:
	ldr	q0, [x19]
	fneg	v0.4s, v0.4s
	str	q0, [x19], 16
	cmp	x19, x0
	bne	.L518
	b	.L685
.L534:
	.cfi_offset 23, -208
	.cfi_offset 24, -200
	.cfi_offset 27, -176
	.cfi_offset 28, -168
	cmp	w4, 0
	ble	.L708
	sub	w27, w2, #1
	stp	x25, x26, [sp, 64]
	.cfi_offset 26, -184
	.cfi_offset 25, -192
	add	x0, x27, 1
	mov	w26, 3
	sub	w26, w26, w2
	add	x26, x26, 1
	lsl	x0, x0, 2
	add	x27, sp, 176
	str	x0, [sp, 136]
	sxtw	x0, w2
	mov	w25, 0
	str	x0, [sp, 144]
	lsl	x0, x26, 2
	mov	w26, 0
	str	x0, [sp, 128]
.L540:
	lsl	w5, w26, 2
	cmp	w20, 0
	ble	.L539
	ldr	x2, [sp, 136]
	add	x0, x27, x5, sxtw 2
	add	x1, x23, x25, sxtw 2
	str	w5, [sp, 120]
	bl	memcpy
	ldr	w5, [sp, 120]
	cmp	w20, 3
	bgt	.L542
.L539:
	ldr	x0, [sp, 144]
	mov	w1, 0
	ldr	x2, [sp, 128]
	add	x0, x0, x5, sxtw
	add	x0, x27, x0, lsl 2
	bl	memset
.L542:
	add	w26, w26, 1
	add	w25, w25, w24
	cmp	w28, w26
	bne	.L540
	b	.L683
.L694:
	add	w11, w0, 4
	b	.L547
.L577:
	.cfi_restore 23
	.cfi_restore 24
	.cfi_restore 25
	.cfi_restore 26
	.cfi_restore 27
	.cfi_restore 28
	mov	w0, 0
	b	.L523
.L552:
	.cfi_offset 23, -208
	.cfi_offset 24, -200
	.cfi_offset 27, -176
	.cfi_offset 28, -168
	cmp	w4, 0
	ble	.L708
	lsl	w10, w6, 1
	mov	w2, 3
	add	w0, w10, w6
	sub	w2, w2, w20
	add	x2, x2, 1
	str	w0, [sp, 172]
	add	w0, w4, w6
	lsl	w8, w6, 3
	str	w0, [sp, 156]
	lsl	w0, w6, 2
	sub	w9, w8, w6
	lsl	x2, x2, 2
	add	x27, sp, 176
	stp	x25, x26, [sp, 64]
	.cfi_offset 26, -184
	.cfi_offset 25, -192
	mov	w25, w6
	str	w0, [sp, 168]
	sxtw	x0, w20
	mov	x26, 0
	str	x0, [sp, 160]
	mov	w0, 0
.L559:
	sxtw	x1, w0
	cmp	w20, 0
	ble	.L695
	ldr	s0, [x23, x26, lsl 2]
	str	s0, [x27, x1, lsl 2]
	cmp	w20, 1
	beq	.L695
	add	w7, w0, 1
	ldr	s0, [x23, w25, sxtw 2]
	str	s0, [x27, w7, sxtw 2]
	cmp	w20, 2
	beq	.L695
	add	w7, w10, w26
	add	w12, w0, 2
	ldr	s0, [x23, w7, sxtw 2]
	str	s0, [x27, w12, sxtw 2]
	cmp	w20, 3
	beq	.L695
	add	w7, w10, w25
	add	w12, w0, 3
	ldr	s0, [x23, w7, sxtw 2]
	str	s0, [x27, w12, sxtw 2]
	cmp	w20, 4
	beq	.L715
	ldr	w3, [sp, 168]
	add	w11, w0, 4
	add	w12, w3, w26
	ldr	s0, [x23, w12, sxtw 2]
	str	s0, [x27, w11, sxtw 2]
	cmp	w20, 5
	beq	.L561
	add	w7, w10, w7
	add	w12, w0, 5
	ldr	s0, [x23, w7, sxtw 2]
	str	s0, [x27, w12, sxtw 2]
	cmp	w20, 6
	beq	.L561
	add	w7, w24, w7
	add	w12, w0, 6
	ldr	s0, [x23, w7, sxtw 2]
	str	s0, [x27, w12, sxtw 2]
	cmp	w20, 7
	beq	.L561
	add	w7, w0, 7
	ldr	s0, [x23, w9, sxtw 2]
	str	s0, [x27, w7, sxtw 2]
	cmp	w20, 8
	beq	.L561
	add	w7, w0, 8
	ldr	s0, [x23, w8, sxtw 2]
	str	s0, [x27, w7, sxtw 2]
	cmp	w20, 9
	beq	.L561
	add	w7, w24, w8
	add	w12, w0, 9
	ldr	s0, [x23, w7, sxtw 2]
	str	s0, [x27, w12, sxtw 2]
	cmp	w20, 10
	beq	.L561
	add	w7, w24, w7
	add	w12, w0, 10
	ldr	s0, [x23, w7, sxtw 2]
	str	s0, [x27, w12, sxtw 2]
	cmp	w20, 11
	beq	.L561
	add	w13, w3, w9
	add	w12, w0, 11
	ldr	s0, [x23, w13, sxtw 2]
	str	s0, [x27, w12, sxtw 2]
	cmp	w20, 12
	beq	.L561
	add	w7, w10, w7
	add	w12, w0, 12
	ldr	s0, [x23, w7, sxtw 2]
	str	s0, [x27, w12, sxtw 2]
	cmp	w20, 13
	beq	.L561
	add	w7, w24, w7
	add	w12, w0, 13
	ldr	s0, [x23, w7, sxtw 2]
	str	s0, [x27, w12, sxtw 2]
	cmp	w20, 14
	beq	.L561
	ldr	w3, [sp, 172]
	add	w12, w0, 14
	add	w13, w3, w13
	ldr	s0, [x23, w13, sxtw 2]
	str	s0, [x27, w12, sxtw 2]
	cmp	w20, 15
	beq	.L561
	add	w7, w10, w7
	add	w0, w0, 15
	ldr	s0, [x23, w7, sxtw 2]
	str	s0, [x27, w0, sxtw 2]
	cmp	w20, 3
	bgt	.L561
.L558:
	ldr	x0, [sp, 160]
	str	x2, [sp, 120]
	str	w10, [sp, 128]
	add	x0, x1, x0
	mov	w1, 0
	str	w9, [sp, 136]
	add	x0, x27, x0, lsl 2
	str	w8, [sp, 144]
	str	w11, [sp, 152]
	bl	memset
	ldr	w10, [sp, 128]
	ldr	w9, [sp, 136]
	ldr	w8, [sp, 144]
	ldr	w11, [sp, 152]
	ldr	x2, [sp, 120]
.L561:
	add	w25, w25, 1
	ldr	w1, [sp, 156]
	add	x26, x26, 1
	mov	w0, w11
	add	w9, w9, 1
	add	w8, w8, 1
	cmp	w25, w1
	bne	.L559
	b	.L683
.L695:
	add	w11, w0, 4
	b	.L558
.L708:
	.cfi_restore 25
	.cfi_restore 26
	add	x27, sp, 176
	b	.L554
.L514:
	.cfi_restore 23
	.cfi_restore 24
	.cfi_restore 27
	.cfi_restore 28
	dup	v8.4s, v1.s[0]
	add	x1, x19, 16
	add	x0, x1, x0, uxtw 4
	.p2align 3,,7
.L517:
	ldr	q0, [x19]
	fmul	v0.4s, v8.4s, v0.4s
	str	q0, [x19], 16
	cmp	x0, x19
	bne	.L517
	b	.L685
.L714:
	.cfi_offset 23, -208
	.cfi_offset 24, -200
	.cfi_offset 25, -192
	.cfi_offset 26, -184
	.cfi_offset 27, -176
	.cfi_offset 28, -168
	add	w11, w0, 4
	b	.L550
.L715:
	add	w11, w0, 4
	b	.L561
.L711:
	.cfi_restore 21
	.cfi_restore 22
	.cfi_restore 23
	.cfi_restore 24
	.cfi_restore 25
	.cfi_restore 26
	.cfi_restore 27
	.cfi_restore 28
	.cfi_restore 72
	.cfi_restore 73
	stp	x21, x22, [sp, 32]
	.cfi_offset 22, -216
	.cfi_offset 21, -224
	stp	x23, x24, [sp, 48]
	.cfi_offset 24, -200
	.cfi_offset 23, -208
	stp	x25, x26, [sp, 64]
	.cfi_offset 26, -184
	.cfi_offset 25, -192
	stp	x27, x28, [sp, 80]
	.cfi_offset 28, -168
	.cfi_offset 27, -176
	stp	d8, d9, [sp, 96]
	.cfi_offset 73, -152
	.cfi_offset 72, -160
	bl	__stack_chk_fail
.L568:
	.cfi_restore 25
	.cfi_restore 26
	mov	w3, w4
	adrp	x1, .LC0
	add	x1, x1, :lo12:.LC0
	mov	w0, 1
	stp	x25, x26, [sp, 64]
	.cfi_offset 26, -184
	.cfi_offset 25, -192
	bl	__printf_chk
	mov	w0, -1
	bl	exit
	.cfi_endproc
.LFE4313:
	.size	gemm_microkernel_ABresident_neon_4x4_fp32, .-gemm_microkernel_ABresident_neon_4x4_fp32
	.align	2
	.p2align 4,,11
	.global	fvtrans_float32_4x4
	.type	fvtrans_float32_4x4, %function
fvtrans_float32_4x4:
.LFB4314:
	.cfi_startproc
	ldr	q3, [x2]
	ldr	q4, [x3]
	ldr	q0, [x0]
	ldr	q1, [x1]
	zip1	v2.2d, v0.2d, v3.2d
	zip2	v0.2d, v0.2d, v3.2d
	zip1	v3.2d, v1.2d, v4.2d
	zip2	v1.2d, v1.2d, v4.2d
	trn1	v4.4s, v2.4s, v3.4s
	trn2	v2.4s, v2.4s, v3.4s
	trn1	v3.4s, v0.4s, v1.4s
	trn2	v0.4s, v0.4s, v1.4s
	str	q4, [x0]
	str	q2, [x1]
	str	q3, [x2]
	str	q0, [x3]
	ret
	.cfi_endproc
.LFE4314:
	.size	fvtrans_float32_4x4, .-fvtrans_float32_4x4
	.align	2
	.p2align 4,,11
	.global	fvtrans_float32_8x8
	.type	fvtrans_float32_8x8, %function
fvtrans_float32_8x8:
.LFB4315:
	.cfi_startproc
	ldr	q3, [x2]
	ldr	q4, [x3]
	ldr	q0, [x0]
	ldr	q1, [x1]
	zip1	v2.2d, v0.2d, v3.2d
	zip2	v0.2d, v0.2d, v3.2d
	zip1	v3.2d, v1.2d, v4.2d
	zip2	v1.2d, v1.2d, v4.2d
	ldp	x10, x9, [sp]
	trn1	v4.4s, v2.4s, v3.4s
	trn2	v2.4s, v2.4s, v3.4s
	trn1	v3.4s, v0.4s, v1.4s
	trn2	v0.4s, v0.4s, v1.4s
	str	q4, [x0]
	ldr	x8, [sp, 16]
	str	q2, [x1]
	ldr	x0, [sp, 24]
	str	q3, [x2]
	str	q0, [x3]
	ldr	q4, [x0]
	ldr	q3, [x8]
	ldr	q1, [x9]
	ldr	q0, [x10]
	ldr	x3, [sp, 32]
	zip1	v2.2d, v0.2d, v3.2d
	zip2	v0.2d, v0.2d, v3.2d
	zip1	v3.2d, v1.2d, v4.2d
	zip2	v1.2d, v1.2d, v4.2d
	ldr	x2, [sp, 40]
	trn1	v4.4s, v2.4s, v3.4s
	trn2	v2.4s, v2.4s, v3.4s
	trn1	v3.4s, v0.4s, v1.4s
	trn2	v0.4s, v0.4s, v1.4s
	str	q4, [x10]
	str	q2, [x9]
	str	q3, [x8]
	str	q0, [x0]
	ldr	q3, [x6]
	ldr	q4, [x7]
	ldr	q0, [x4]
	ldr	q1, [x5]
	zip1	v2.2d, v0.2d, v3.2d
	zip2	v0.2d, v0.2d, v3.2d
	zip1	v3.2d, v1.2d, v4.2d
	zip2	v1.2d, v1.2d, v4.2d
	ldr	x1, [sp, 48]
	trn1	v4.4s, v2.4s, v3.4s
	trn2	v2.4s, v2.4s, v3.4s
	trn1	v3.4s, v0.4s, v1.4s
	trn2	v0.4s, v0.4s, v1.4s
	ldr	x0, [sp, 56]
	str	q4, [x4]
	str	q2, [x5]
	str	q3, [x6]
	str	q0, [x7]
	ldr	q0, [x3]
	ldr	q4, [x0]
	ldr	q3, [x1]
	ldr	q1, [x2]
	zip1	v2.2d, v0.2d, v3.2d
	zip2	v0.2d, v0.2d, v3.2d
	zip1	v3.2d, v1.2d, v4.2d
	zip2	v1.2d, v1.2d, v4.2d
	trn1	v4.4s, v2.4s, v3.4s
	trn2	v2.4s, v2.4s, v3.4s
	trn1	v3.4s, v0.4s, v1.4s
	trn2	v0.4s, v0.4s, v1.4s
	str	q4, [x3]
	str	q2, [x2]
	str	q3, [x1]
	str	q0, [x0]
	ret
	.cfi_endproc
.LFE4315:
	.size	fvtrans_float32_8x8, .-fvtrans_float32_8x8
	.align	2
	.p2align 4,,11
	.global	dot
	.type	dot, %function
dot:
.LFB4316:
	.cfi_startproc
	movi	v2.4s, 0
	fmla	v2.4s, v0.4s, v1.4s
	faddp	v2.4s, v2.4s, v2.4s
	faddp	v2.4s, v2.4s, v2.4s
	dup	s0, v2.s[0]
	ret
	.cfi_endproc
.LFE4316:
	.size	dot, .-dot
	.ident	"GCC: (Ubuntu 10.3.0-1ubuntu1~18.04~1) 10.3.0"
	.section	.note.GNU-stack,"",@progbits

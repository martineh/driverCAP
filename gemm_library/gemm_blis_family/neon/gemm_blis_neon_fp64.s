	.text
	.file	"gemm_blis_neon_fp64.c"
	.globl	gemm_microkernel_Cresident_neon_4x4_prefetch_fp64 // -- Begin function gemm_microkernel_Cresident_neon_4x4_prefetch_fp64
	.p2align	2
	.type	gemm_microkernel_Cresident_neon_4x4_prefetch_fp64,@function
gemm_microkernel_Cresident_neon_4x4_prefetch_fp64: // @gemm_microkernel_Cresident_neon_4x4_prefetch_fp64
// %bb.0:
	str	x28, [sp, #-80]!        // 8-byte Folded Spill
	stp	x24, x23, [sp, #16]     // 8-byte Folded Spill
	stp	x22, x21, [sp, #32]     // 8-byte Folded Spill
	stp	x20, x19, [sp, #48]     // 8-byte Folded Spill
	stp	x29, x30, [sp, #64]     // 8-byte Folded Spill
	add	x29, sp, #64            // =64
	sub	sp, sp, #512            // =512
	mov	x19, x6
                                        // kill: def %w7 killed %w7 def %x7
                                        // kill: def %d1 killed %d1 def %q1
                                        // kill: def %d0 killed %d0 def %q0
	cbz	w3, .LBB0_36
// %bb.1:
	and	w8, w0, #0xff
	cmp	w8, #67                 // =67
	csel	x9, x5, x4, eq
	csel	x10, x4, x5, eq
	fcmp	d0, #0.0
	b.eq	.LBB0_5
// %bb.2:
	ldp	q21, q18, [x10]
	ldp	q20, q19, [x9]
	cmp	w3, #2                  // =2
	movi	v2.2d, #0000000000000000
	b.lt	.LBB0_11
// %bb.3:
	sub	w8, w3, #1              // =1
	add	x9, x9, #80             // =80
	add	x10, x10, #80           // =80
	movi	v7.2d, #0000000000000000
	movi	v17.2d, #0000000000000000
	movi	v16.2d, #0000000000000000
	movi	v6.2d, #0000000000000000
	movi	v5.2d, #0000000000000000
	movi	v4.2d, #0000000000000000
	movi	v3.2d, #0000000000000000
	movi	v2.2d, #0000000000000000
.LBB0_4:                                // =>This Inner Loop Header: Depth=1
	fmla	v7.2d, v21.2d, v20.d[0]
	fmla	v17.2d, v21.2d, v20.d[1]
	fmla	v16.2d, v21.2d, v19.d[0]
	fmla	v6.2d, v21.2d, v19.d[1]
	fmla	v5.2d, v18.2d, v20.d[0]
	fmla	v4.2d, v18.2d, v20.d[1]
	fmla	v3.2d, v18.2d, v19.d[0]
	fmla	v2.2d, v18.2d, v19.d[1]
	ldp	q21, q18, [x10, #-16]
	ldp	q20, q19, [x9, #-16]
	add	x9, x9, #64             // =64
	subs	w8, w8, #1              // =1
	add	x10, x10, #64           // =64
	b.ne	.LBB0_4
	b	.LBB0_12
.LBB0_5:
	movi	v2.2d, #0000000000000000
	movi	v3.2d, #0000000000000000
	movi	v4.2d, #0000000000000000
	movi	v5.2d, #0000000000000000
	movi	v6.2d, #0000000000000000
	movi	v16.2d, #0000000000000000
	movi	v17.2d, #0000000000000000
	movi	v7.2d, #0000000000000000
	cmp	w1, #8                  // =8
	b.lt	.LBB0_17
.LBB0_6:
	cmp	w2, #7                  // =7
	b.le	.LBB0_17
// %bb.7:
	cmp	w1, #8                  // =8
	b.ne	.LBB0_53
// %bb.8:
	cmp	w2, #8                  // =8
	b.ne	.LBB0_53
// %bb.9:
	fcmp	d1, #0.0
	b.eq	.LBB0_34
// %bb.10:
	add	x8, x19, w7, sxtw #3
	ldp	q0, q20, [x19]
	mov	x10, x8
	sxtw	x11, w7
	ldr	q18, [x10], #16
	lsl	w12, w7, #1
	add	x13, x11, w7, sxtw #1
	add	x9, x19, w7, sxtw #4
	add	w11, w12, #2            // =2
	add	x12, x19, x13, lsl #3
	add	w13, w13, #2            // =2
	ldr	q19, [x9]
	add	x11, x19, w11, sxtw #3
	ldr	q21, [x12]
	add	x13, x19, w13, sxtw #3
	fmul	v0.2d, v0.2d, v1.d[0]
	ldr	q22, [x11]
	ldr	q23, [x13]
	fadd	v7.2d, v7.2d, v0.2d
	ldr	q0, [x10]
	fmul	v18.2d, v18.2d, v1.d[0]
	fmul	v19.2d, v19.2d, v1.d[0]
	fadd	v17.2d, v17.2d, v18.2d
	fmul	v18.2d, v21.2d, v1.d[0]
	fmul	v20.2d, v20.2d, v1.d[0]
	fadd	v16.2d, v16.2d, v19.2d
	fmul	v19.2d, v22.2d, v1.d[0]
	fadd	v6.2d, v6.2d, v18.2d
	fmul	v18.2d, v23.2d, v1.d[0]
	fmul	v0.2d, v0.2d, v1.d[0]
	fadd	v5.2d, v5.2d, v20.2d
	fadd	v3.2d, v3.2d, v19.2d
	fadd	v4.2d, v4.2d, v0.2d
	fadd	v2.2d, v2.2d, v18.2d
	b	.LBB0_35
.LBB0_11:
	movi	v3.2d, #0000000000000000
	movi	v4.2d, #0000000000000000
	movi	v5.2d, #0000000000000000
	movi	v6.2d, #0000000000000000
	movi	v16.2d, #0000000000000000
	movi	v17.2d, #0000000000000000
	movi	v7.2d, #0000000000000000
.LBB0_12:
	fmla	v7.2d, v21.2d, v20.d[0]
	fmla	v17.2d, v21.2d, v20.d[1]
	fmla	v5.2d, v18.2d, v20.d[0]
	fmla	v4.2d, v18.2d, v20.d[1]
	fmov	d20, #-1.00000000
	fmla	v16.2d, v21.2d, v19.d[0]
	fmla	v6.2d, v21.2d, v19.d[1]
	fmla	v3.2d, v18.2d, v19.d[0]
	fcmp	d0, d20
	fmla	v2.2d, v18.2d, v19.d[1]
	b.ne	.LBB0_14
// %bb.13:
	fneg	v7.2d, v7.2d
	fneg	v17.2d, v17.2d
	fneg	v16.2d, v16.2d
	fneg	v6.2d, v6.2d
	fneg	v5.2d, v5.2d
	fneg	v4.2d, v4.2d
	fneg	v3.2d, v3.2d
	fneg	v2.2d, v2.2d
	cmp	w1, #8                  // =8
	b.ge	.LBB0_6
	b	.LBB0_17
.LBB0_14:
	fmov	d18, #1.00000000
	fcmp	d0, d18
	b.eq	.LBB0_16
// %bb.15:
	fmul	v7.2d, v7.2d, v0.d[0]
	fmul	v17.2d, v17.2d, v0.d[0]
	fmul	v16.2d, v16.2d, v0.d[0]
	fmul	v6.2d, v6.2d, v0.d[0]
	fmul	v5.2d, v5.2d, v0.d[0]
	fmul	v4.2d, v4.2d, v0.d[0]
	fmul	v3.2d, v3.2d, v0.d[0]
	fmul	v2.2d, v2.2d, v0.d[0]
.LBB0_16:
	cmp	w1, #8                  // =8
	b.ge	.LBB0_6
.LBB0_17:
	cmp	w2, #0                  // =0
	cset	w8, gt
	cmp	w1, #0                  // =0
	cset	w9, gt
	fcmp	d1, #0.0
	and	w8, w8, w9
	and	w9, w0, #0xff
	stp	q7, q5, [sp]
	stp	q17, q4, [sp, #64]
	stp	q16, q3, [sp, #128]
	stp	q6, q2, [sp, #192]
	b.eq	.LBB0_30
// %bb.18:
	cmp	w9, #67                 // =67
	b.ne	.LBB0_37
// %bb.19:
	cbz	w8, .LBB0_36
// %bb.20:
	mov	w10, w1
	sub	x14, x10, #1            // =1
	mov	x8, xzr
	mov	x9, xzr
	mov	w11, w2
	dup	v0.2d, v1.d[0]
	add	x12, x19, #16           // =16
	sbfiz	x13, x7, #3, #32
	and	x15, x10, #0xfffffffc
	mov	x16, sp
	lsr	x17, x14, #32
	b	.LBB0_26
.LBB0_21:                               //   in Loop: Header=BB0_26 Depth=1
	lsl	w18, w9, #3
	cmn	w18, w14
	mov	x18, xzr
	b.hs	.LBB0_28
// %bb.22:                              //   in Loop: Header=BB0_26 Depth=1
	cbnz	x17, .LBB0_28
// %bb.23:                              //   in Loop: Header=BB0_26 Depth=1
	mov	x18, x15
	mov	x0, x8
	mov	x2, x12
.LBB0_24:                               //   Parent Loop BB0_26 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	and	x3, x0, #0xfffffffc
	ldp	q2, q3, [x2, #-16]
	add	x3, x16, x3, lsl #3
	ldp	q4, q5, [x3]
	subs	x18, x18, #4            // =4
	fmul	v2.2d, v2.2d, v0.2d
	fmul	v3.2d, v3.2d, v0.2d
	fadd	v2.2d, v2.2d, v4.2d
	fadd	v3.2d, v3.2d, v5.2d
	stp	q2, q3, [x2, #-16]
	add	x2, x2, #32             // =32
	add	x0, x0, #4              // =4
	b.ne	.LBB0_24
// %bb.25:                              //   in Loop: Header=BB0_26 Depth=1
	cmp	x15, x10
	mov	x18, x15
	b.ne	.LBB0_28
	b	.LBB0_29
.LBB0_26:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_24 Depth 2
                                        //     Child Loop BB0_28 Depth 2
	cmp	w1, #4                  // =4
	b.hs	.LBB0_21
// %bb.27:                              //   in Loop: Header=BB0_26 Depth=1
	mov	x18, xzr
.LBB0_28:                               //   Parent Loop BB0_26 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	lsl	x0, x18, #3
	add	w2, w8, w18
	ldr	d2, [x19, x0]
	ldr	d3, [x16, w2, uxtw #3]
	add	x18, x18, #1            // =1
	cmp	x10, x18
	fmul	d2, d2, d1
	fadd	d2, d2, d3
	str	d2, [x19, x0]
	b.ne	.LBB0_28
.LBB0_29:                               //   in Loop: Header=BB0_26 Depth=1
	add	x9, x9, #1              // =1
	add	x12, x12, x13
	add	x8, x8, #8              // =8
	cmp	x9, x11
	add	x19, x19, x13
	b.ne	.LBB0_26
	b	.LBB0_36
.LBB0_30:
	cmp	w9, #67                 // =67
	b.ne	.LBB0_42
// %bb.31:
	cbz	w8, .LBB0_36
// %bb.32:
	mov	w8, w1
	mov	x21, xzr
	mov	w22, w2
	sbfiz	x23, x7, #3, #32
	lsl	x20, x8, #3
	mov	x24, sp
.LBB0_33:                               // =>This Inner Loop Header: Depth=1
	and	x8, x21, #0xfffffff8
	add	x1, x24, x8, lsl #3
	mov	x0, x19
	mov	x2, x20
	bl	memcpy
	add	x21, x21, #8            // =8
	subs	x22, x22, #1            // =1
	add	x19, x19, x23
	b.ne	.LBB0_33
	b	.LBB0_36
.LBB0_34:
	sxtw	x10, w7
	lsl	w11, w7, #1
	add	x13, x10, w7, sxtw #1
	add	x8, x19, w7, sxtw #3
	add	w11, w11, #2            // =2
	add	x12, x19, x13, lsl #3
	add	w13, w13, #2            // =2
	add	x9, x19, w7, sxtw #4
	add	x10, x8, #16            // =16
	add	x11, x19, w11, sxtw #3
	add	x13, x19, w13, sxtw #3
.LBB0_35:
	str	q7, [x19]
	str	q17, [x8]
	str	q16, [x9]
	str	q6, [x12]
	str	q5, [x19, #16]
	str	q4, [x10]
	str	q3, [x11]
	str	q2, [x13]
.LBB0_36:
	add	sp, sp, #512            // =512
	ldp	x29, x30, [sp, #64]     // 8-byte Folded Reload
	ldp	x20, x19, [sp, #48]     // 8-byte Folded Reload
	ldp	x22, x21, [sp, #32]     // 8-byte Folded Reload
	ldp	x24, x23, [sp, #16]     // 8-byte Folded Reload
	ldr	x28, [sp], #80          // 8-byte Folded Reload
	ret
.LBB0_37:
	cbz	w8, .LBB0_36
// %bb.38:
	mov	x8, xzr
	mov	w9, w1
	mov	w10, w2
	sbfiz	x11, x7, #3, #32
	mov	x12, sp
.LBB0_39:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_40 Depth 2
	mov	x13, x9
	mov	x14, x19
	mov	x15, x8
.LBB0_40:                               //   Parent Loop BB0_39 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	d0, [x14]
	ldr	d2, [x12, w15, uxtw #3]
	add	x15, x15, #8            // =8
	subs	x13, x13, #1            // =1
	fmul	d0, d0, d1
	fadd	d0, d0, d2
	str	d0, [x14]
	add	x14, x14, x11
	b.ne	.LBB0_40
// %bb.41:                              //   in Loop: Header=BB0_39 Depth=1
	add	x8, x8, #1              // =1
	cmp	x8, x10
	add	x19, x19, #8            // =8
	b.ne	.LBB0_39
	b	.LBB0_36
.LBB0_42:
	cbz	w8, .LBB0_36
// %bb.43:
	mov	w10, w1
	sub	x0, x10, #1             // =1
	and	x17, x0, #0xffffffff
	lsl	x17, x17, #3
	cmp	xzr, x17, lsr #32
	mov	x8, xzr
	sxtw	x9, w7
	mov	w11, w2
	sbfiz	x12, x7, #3, #32
	sbfiz	x13, x7, #4, #32
	orr	w14, wzr, #0x8
	mov	x15, sp
	and	x16, x10, #0xfffffffe
	cset	w18, ne
	lsr	x0, x0, #32
	mov	x2, x19
	b	.LBB0_47
.LBB0_44:                               //   in Loop: Header=BB0_47 Depth=1
	mov	x3, x16
	mov	x4, x2
	mov	x5, x14
.LBB0_45:                               //   Parent Loop BB0_47 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	w6, w5, #8              // =8
	ldr	x6, [x15, w6, uxtw #3]
	ldr	x7, [x15, w5, uxtw #3]
	add	x5, x5, #16             // =16
	subs	x3, x3, #2              // =2
	str	x6, [x4]
	str	x7, [x4, x12]
	add	x4, x4, x13
	b.ne	.LBB0_45
// %bb.46:                              //   in Loop: Header=BB0_47 Depth=1
	cmp	x16, x10
	mov	x5, x16
	b.eq	.LBB0_52
	b	.LBB0_50
.LBB0_47:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_45 Depth 2
                                        //     Child Loop BB0_51 Depth 2
	cmp	w1, #2                  // =2
	b.lo	.LBB0_49
// %bb.48:                              //   in Loop: Header=BB0_47 Depth=1
	cmp	x0, #0                  // =0
	cset	w3, ne
	cmn	w17, w8
	cset	w4, hs
	orr	w3, w4, w3
	orr	w3, w3, w18
	tbz	w3, #0, .LBB0_44
.LBB0_49:                               //   in Loop: Header=BB0_47 Depth=1
	mov	x5, xzr
.LBB0_50:                               //   in Loop: Header=BB0_47 Depth=1
	madd	x4, x9, x5, x8
	add	x3, x8, x5, lsl #3
	add	x4, x19, x4, lsl #3
	sub	x5, x10, x5
.LBB0_51:                               //   Parent Loop BB0_47 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	x6, [x15, w3, uxtw #3]
	add	x3, x3, #8              // =8
	subs	x5, x5, #1              // =1
	str	x6, [x4]
	add	x4, x4, x12
	b.ne	.LBB0_51
.LBB0_52:                               //   in Loop: Header=BB0_47 Depth=1
	add	x8, x8, #1              // =1
	add	x14, x14, #1            // =1
	cmp	x8, x11
	add	x2, x2, #8              // =8
	b.ne	.LBB0_47
	b	.LBB0_36
.LBB0_53:
	adrp	x0, .L.str
	add	x0, x0, :lo12:.L.str
	bl	printf
	mov	w0, #-1
	bl	exit
.Lfunc_end0:
	.size	gemm_microkernel_Cresident_neon_4x4_prefetch_fp64, .Lfunc_end0-gemm_microkernel_Cresident_neon_4x4_prefetch_fp64
                                        // -- End function
	.globl	gemm_microkernel_Cresident_neon_8x8_prefetch_fp64 // -- Begin function gemm_microkernel_Cresident_neon_8x8_prefetch_fp64
	.p2align	2
	.type	gemm_microkernel_Cresident_neon_8x8_prefetch_fp64,@function
gemm_microkernel_Cresident_neon_8x8_prefetch_fp64: // @gemm_microkernel_Cresident_neon_8x8_prefetch_fp64
// %bb.0:
	stp	x29, x30, [sp, #-16]!   // 8-byte Folded Spill
	adrp	x0, .L.str.1
	add	x0, x0, :lo12:.L.str.1
	mov	x29, sp
	bl	printf
	mov	w0, #-1
	bl	exit
.Lfunc_end1:
	.size	gemm_microkernel_Cresident_neon_8x8_prefetch_fp64, .Lfunc_end1-gemm_microkernel_Cresident_neon_8x8_prefetch_fp64
                                        // -- End function
	.globl	gemm_microkernel_ABresident_neon_4x4_prefetch_fp64 // -- Begin function gemm_microkernel_ABresident_neon_4x4_prefetch_fp64
	.p2align	2
	.type	gemm_microkernel_ABresident_neon_4x4_prefetch_fp64,@function
gemm_microkernel_ABresident_neon_4x4_prefetch_fp64: // @gemm_microkernel_ABresident_neon_4x4_prefetch_fp64
// %bb.0:
	stp	x29, x30, [sp, #-16]!   // 8-byte Folded Spill
	adrp	x0, .L.str.1
	add	x0, x0, :lo12:.L.str.1
	mov	w1, w2
	mov	w2, w3
	mov	x29, sp
	bl	printf
	mov	w0, #-1
	bl	exit
.Lfunc_end2:
	.size	gemm_microkernel_ABresident_neon_4x4_prefetch_fp64, .Lfunc_end2-gemm_microkernel_ABresident_neon_4x4_prefetch_fp64
                                        // -- End function
	.type	.L.str,@object          // @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"Error: Incorrect use of 4x4 micro-kernel with %d x %d block\n"
	.size	.L.str, 61

	.type	.L.str.1,@object        // @.str.1
.L.str.1:
	.asciz	"Error: not implemented %d %d\n"
	.size	.L.str.1, 30


	.ident	"clang version 6.0.0-1ubuntu2 (tags/RELEASE_600/final)"
	.section	".note.GNU-stack","",@progbits

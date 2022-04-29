.data
	a: .space 4

.text
	.global _start
_start:

	.global odd
	b _afterodd
odd:
	ldr r4, =a
	ldr r0, [r4]

	ldr r1, =a
	ldr r0, [r1]
	cmp r0, #0
	beq _afterifa8214503179697027083
	ldr r4, =a
	ldr r0, [r4]
	mov r1, #1
	sub r0, r1
	ldr r1, =a
	str r0, [r1]
_afterifa8214503179697027083:
_afterodd:
	pop {r4, r5, r6, r7, PC}
	push {r4, r5, r6, r7, lr}
	b odd
	mov r0, #1

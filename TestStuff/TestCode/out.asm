.data
	a: .space 4
	b: .space 4

.text
	.global _start
_start:

	.global odd
	b _afterodd
odd:
	push {r4, r5, r6, r7, lr}
	ldr r4, =a
	str r0, [r4]

	ldr r1, =a
	ldr r0, [r1]
	cmp r0, #0
	beq _afterifa_1599606244698471856
	ldr r4, =a
	ldr r0, [r4]
	mov r1, #1
	sub r0, r1
	ldr r1, =a
	str r0, [r1]
	ldr r4, =a
	ldr r0, [r4]
	bl even
	pop {r4, r5, r6, r7, PC}
_afterifa_1599606244698471856:

	ldr r1, =a
	ldr r0, [r1]
	cmp r0, #0
	bne _afterifna_12328836034767429347
	mov r0, #0
	pop {r4, r5, r6, r7, PC}
_afterifna_12328836034767429347:
	pop {r4, r5, r6, r7, PC}

_afterodd:

	.global even
	b _aftereven
even:
	push {r4, r5, r6, r7, lr}
	ldr r4, =a
	str r0, [r4]

	ldr r1, =a
	ldr r0, [r1]
	cmp r0, #0
	beq _afterifa_9162913032136070979
	ldr r4, =a
	ldr r0, [r4]
	mov r1, #1
	sub r0, r1
	ldr r1, =a
	str r0, [r1]
	ldr r4, =a
	ldr r0, [r4]
	bl odd
	pop {r4, r5, r6, r7, PC}
_afterifa_9162913032136070979:

	ldr r1, =a
	ldr r0, [r1]
	cmp r0, #0
	bne _afterifna_7980942264373446747
	mov r0, #1
	pop {r4, r5, r6, r7, PC}
_afterifna_7980942264373446747:
	pop {r4, r5, r6, r7, PC}

_aftereven:

	.global sommig
	b _aftersommig
sommig:
	push {r4, r5, r6, r7, lr}
	ldr r4, =a
	str r0, [r4]
	ldr r4, =b
	str r1, [r4]

	ldr r1, =a
	ldr r0, [r1]
	cmp r0, #0
	beq _afterwhilea
_startwhilea:
	ldr r4, =b
	ldr r0, [r4]
	mov r1, #1
	add r0, r1
	ldr r1, =b
	str r0, [r1]
	ldr r4, =a
	ldr r0, [r4]
	mov r1, #1
	sub r0, r1
	ldr r1, =a
	str r0, [r1]
	ldr r1, =a
	ldr r0, [r1]
	cmp r0, #0
	bne _startwhilea

_afterwhilea:
	ldr r1, =b
	ldr r0, [r1]
	pop {r4, r5, r6, r7, PC}
	pop {r4, r5, r6, r7, PC}

_aftersommig:
	mov r0, #1

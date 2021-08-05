class architecture:
    SAFEINDEX = 0
    REGISTERS = []

class x86(architecture):
    R0 = "eax"
    R1 = "ebx"
    R2 = "ecx"
    R3 = "edx"
    R4 = "ebp"
    R5 = "esi"
    R6 = "edi"
    R7 = "esp"
    R8 = "r8d"
    R9 = "r9d"
    R10 = "r10d"
    R11 = "r11d"
    R12 = "r12d"
    R13 = "r13d"
    R14 = "r14d"
    R15 = "r15d"

    SAFEINDEX = 4
    REGISTERS = [R0,R1,R2,R3,R4,R5,R6,R7,R8,R9,R10,R11,R12,R13,R14,R15]

class cortex(architecture):
    R0 = "r0"
    R1 = "r1"
    R2 = "r2"
    R3 = "r3"
    R4 = "r4"
    R5 = "r5"
    R6 = "r6"
    R7 = "r7"
    R8 = "r8"
    R9 = "r9"
    R10 = "r10"
    R11 = "r11"
    R12 = "r12"

    SAFEINDEX = 4
    REGISTERS = [R0,R1,R2,R3,R4,R5,R6,R7,R8,R9,R10,R11,R12]

section .data
a dw 0
alen equ $ - a
b dw 0
blen equ $ - b

section .text
global _start
_start:
mov eax, 1
mov [a], eax
function:
mov eax, mov eax, a
mov ebx, 2
section .text
mov eax, a
mov ebx, 2
add eax, ebx

mov [a], eax
pop {ebp, esi, edi, esp, r8d, r9d, r10d, r11d, r12d, r13d, r14d, r15d, pc}
mov eax, [a]
cmp eax, 0
je _afterifa

mov eax, 2
mov [b], eax

_afterifa:

mov eax, 1
int 0x80
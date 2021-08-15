section	.data

msg	db	'Hello, world!',0xa	;our dear string
len	equ	$ - msg			;length of our dear string
a dw 0
alen equ $ - a

section	.text
	global _start       ;must be declared for using gcc
	; global _afterifa

_start:                     ;tell linker entry point
    mov eax, 100
    mov [a], eax
    mov eax, 0
    
;     section .data
;     b dw 0
;     blen equ $ - b
;     section .text
    
;     mov eax, [a]
;     cmp eax, 0
;     je _afterifa
    
;     mov eax, 2
;     mov [b], eax

; _afterifa:
    mov	edx, len    ;message length
	mov	ecx, msg    ;message to write
	mov	ebx, 1	    ;file descriptor (stdout)
	mov	eax, 4	    ;system call number (sys_write)
	int	0x80        ;call kernel
	mov	eax, 1	    ;system call number (sys_exit)
	int	0x80        ;call kernel

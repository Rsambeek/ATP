sudo rm -rf test test.o
nasm -f elf64 test.asm
ld -s -o test test.o
./test
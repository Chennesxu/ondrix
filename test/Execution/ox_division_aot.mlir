// RUN: ondrix-compile %S/../Frontend/Inputs/q15_division.ox --emit=llvmir > %t.ll
// RUN: ondrix-compile %S/../Frontend/Inputs/q15_division.ox --emit=c-header > %t.h
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc -include %t.h %S/Inputs/ox_division_aot.c %t.o -o %t
// RUN: %t

// RUN: ondrix-compile %S/../Frontend/Inputs/q15_infix_precedence.ox --emit=llvmir > %t.infix.ll
// RUN: ondrix-compile %S/../Frontend/Inputs/q15_infix_precedence.ox --emit=c-header > %t.infix.h
// RUN: ondrix-compile %S/../Frontend/Inputs/q15_infix_calls.ox --emit=llvmir > %t.calls.ll
// RUN: ondrix-compile %S/../Frontend/Inputs/q15_infix_calls.ox --emit=c-header > %t.calls.h
// RUN: llc -relocation-model=pic -filetype=obj %t.infix.ll -o %t.infix.o
// RUN: llc -relocation-model=pic -filetype=obj %t.calls.ll -o %t.calls.o
// RUN: cc -include %t.infix.h -include %t.calls.h %S/Inputs/ox_infix_aot.c %t.infix.o %t.calls.o -o %t
// RUN: %t

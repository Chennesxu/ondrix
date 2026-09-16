// RUN: ondrix-compile %S/../Frontend/Inputs/q15_dot.ox --emit=llvmir > %t.dot.ll
// RUN: ondrix-compile %S/../Frontend/Inputs/q15_dot.ox --emit=c-header > %t.dot.h
// RUN: ondrix-compile %S/../Frontend/Inputs/q15_fir_constexpr.ox --emit=llvmir > %t.fir.ll
// RUN: ondrix-compile %S/../Frontend/Inputs/q15_fir_constexpr.ox --emit=c-header > %t.fir.h
// RUN: llc -relocation-model=pic -filetype=obj %t.dot.ll -o %t.dot.o
// RUN: llc -relocation-model=pic -filetype=obj %t.fir.ll -o %t.fir.o
// RUN: cc -include %t.dot.h -include %t.fir.h %S/Inputs/ox_c_entry_aot.c %t.dot.o %t.fir.o -o %t
// RUN: %t

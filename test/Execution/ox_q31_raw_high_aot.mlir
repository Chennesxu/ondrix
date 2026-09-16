// RUN: ondrix-compile %S/../Frontend/Inputs/q31_dot_raw_high.ox --emit=llvmir > %t.dot.ll
// RUN: ondrix-compile %S/../Frontend/Inputs/q31_dot_raw_high.ox --emit=c-header > %t.dot.h
// RUN: ondrix-compile %S/../Frontend/Inputs/q31_fir_raw_high.ox --emit=llvmir > %t.fir.ll
// RUN: ondrix-compile %S/../Frontend/Inputs/q31_fir_raw_high.ox --emit=c-header > %t.fir.h
// RUN: llc -relocation-model=pic -filetype=obj %t.dot.ll -o %t.dot.o
// RUN: llc -relocation-model=pic -filetype=obj %t.fir.ll -o %t.fir.o
// RUN: cc -include %t.dot.h -include %t.fir.h %S/Inputs/ox_q31_raw_high_aot.c %t.dot.o %t.fir.o -o %t
// RUN: %t

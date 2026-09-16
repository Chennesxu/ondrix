// RUN: ondrix-compile %S/../Frontend/Inputs/q31_dct8_raw_high.ox --emit=llvmir > %t.dct8.ll
// RUN: ondrix-compile %S/../Frontend/Inputs/q31_dct8_raw_high.ox --emit=c-header > %t.dct8.h
// RUN: ondrix-compile %S/../Frontend/Inputs/q31_dct64_raw_high.ox --emit=llvmir > %t.dct64.ll
// RUN: ondrix-compile %S/../Frontend/Inputs/q31_dct64_raw_high.ox --emit=c-header > %t.dct64.h
// RUN: llc -relocation-model=pic -filetype=obj %t.dct8.ll -o %t.dct8.o
// RUN: llc -relocation-model=pic -filetype=obj %t.dct64.ll -o %t.dct64.o
// RUN: cc -include %t.dct8.h -include %t.dct64.h %S/Inputs/ox_q31_dct_raw_high_aot.c %t.dct8.o %t.dct64.o -lm -o %t
// RUN: %t

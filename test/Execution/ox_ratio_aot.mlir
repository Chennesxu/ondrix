// RUN: ondrix-compile %S/../Frontend/Inputs/q15_ratio.ox --emit=llvmir > %t.ratio.ll
// RUN: ondrix-compile %S/../Frontend/Inputs/q15_ratio.ox --emit=c-header > %t.ratio.h
// RUN: ondrix-compile %S/../Frontend/Inputs/q15_ratio_saturate.ox --emit=llvmir > %t.saturate.ll
// RUN: ondrix-compile %S/../Frontend/Inputs/q15_ratio_saturate.ox --emit=c-header > %t.saturate.h
// RUN: ondrix-compile %S/../Frontend/Inputs/q31_ratio.ox --emit=llvmir > %t.wide.ll
// RUN: ondrix-compile %S/../Frontend/Inputs/q31_ratio.ox --emit=c-header > %t.wide.h
// RUN: llc -relocation-model=pic -filetype=obj %t.ratio.ll -o %t.ratio.o
// RUN: llc -relocation-model=pic -filetype=obj %t.saturate.ll -o %t.saturate.o
// RUN: llc -relocation-model=pic -filetype=obj %t.wide.ll -o %t.wide.o
// RUN: cc -include %t.ratio.h -include %t.saturate.h -include %t.wide.h %S/Inputs/ox_ratio_aot.c %t.ratio.o %t.saturate.o %t.wide.o -o %t
// RUN: %t

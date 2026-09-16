// RUN: ondrix-compile %S/../Frontend/Inputs/q15_widen.ox --emit=llvmir > %t.widen.ll
// RUN: ondrix-compile %S/../Frontend/Inputs/q15_widen.ox --emit=c-header > %t.widen.h
// RUN: ondrix-compile %S/../Frontend/Inputs/q31_narrow.ox --emit=llvmir > %t.narrow.ll
// RUN: ondrix-compile %S/../Frontend/Inputs/q31_narrow.ox --emit=c-header > %t.narrow.h
// RUN: ondrix-compile %S/../Frontend/Inputs/q31_narrow_even.ox --emit=llvmir > %t.even.ll
// RUN: ondrix-compile %S/../Frontend/Inputs/q31_narrow_even.ox --emit=c-header > %t.even.h
// RUN: ondrix-compile %S/../Frontend/Inputs/q15_conversion_chain.ox --emit=llvmir > %t.chain.ll
// RUN: ondrix-compile %S/../Frontend/Inputs/q15_conversion_chain.ox --emit=c-header > %t.chain.h
// RUN: llc -relocation-model=pic -filetype=obj %t.widen.ll -o %t.widen.o
// RUN: llc -relocation-model=pic -filetype=obj %t.narrow.ll -o %t.narrow.o
// RUN: llc -relocation-model=pic -filetype=obj %t.even.ll -o %t.even.o
// RUN: llc -relocation-model=pic -filetype=obj %t.chain.ll -o %t.chain.o
// RUN: cc -include %t.widen.h -include %t.narrow.h -include %t.even.h -include %t.chain.h %S/Inputs/ox_conversion_aot.c %t.widen.o %t.narrow.o %t.even.o %t.chain.o -o %t
// RUN: %t

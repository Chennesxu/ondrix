// RUN: ondrix-compile %S/../Frontend/Inputs/q15_dot.ox --emit=llvmir > %t.dot.ll
// RUN: ondrix-compile %S/../Frontend/Inputs/q15_dot.ox --emit=c-header > %t.dot.h
// RUN: ondrix-compile %S/../Frontend/Inputs/q15_fir_constexpr.ox --emit=llvmir > %t.fir.ll
// RUN: ondrix-compile %S/../Frontend/Inputs/q15_fir_constexpr.ox --emit=c-header > %t.fir.h
// RUN: ondrix-compile %S/../Frontend/Inputs/q15_multi_use_binding.ox --emit=llvmir > %t.chain.ll
// RUN: ondrix-compile %S/../Frontend/Inputs/q15_multi_use_binding.ox --emit=c-header > %t.chain.h
// RUN: ondrix-compile %S/../Frontend/Inputs/q15_matmul_floor.ox --emit=llvmir > %t.matmul.ll
// RUN: ondrix-compile %S/../Frontend/Inputs/q15_matmul_floor.ox --emit=c-header > %t.matmul.h
// RUN: llc -relocation-model=pic -filetype=obj %t.dot.ll -o %t.dot.o
// RUN: llc -relocation-model=pic -filetype=obj %t.fir.ll -o %t.fir.o
// RUN: llc -relocation-model=pic -filetype=obj %t.chain.ll -o %t.chain.o
// RUN: llc -relocation-model=pic -filetype=obj %t.matmul.ll -o %t.matmul.o
// RUN: cc -include %t.dot.h -include %t.fir.h -include %t.chain.h -include %t.matmul.h %S/Inputs/ox_c_entry_aot.c %t.dot.o %t.fir.o %t.chain.o %t.matmul.o -o %t
// RUN: %t

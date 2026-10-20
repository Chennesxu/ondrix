// RUN: ondrix-compile %S/Inputs/ox_library_blend.ox --emit=llvmir --llvm-opt-level=0 > %t.blend.ll
// RUN: ondrix-compile %S/Inputs/ox_library_blend.ox --emit=c-header > %t.blend.h
// RUN: llc -relocation-model=pic -filetype=obj %t.blend.ll -o %t.blend.o
// RUN: ondrix-compile %S/Inputs/ox_library_spread.ox --emit=llvmir --llvm-opt-level=0 > %t.spread.ll
// RUN: ondrix-compile %S/Inputs/ox_library_spread.ox --emit=c-header > %t.spread.h
// RUN: llc -relocation-model=pic -filetype=obj %t.spread.ll -o %t.spread.o
// RUN: ondrix-compile %S/Inputs/ox_library_above.ox --emit=llvmir --llvm-opt-level=0 > %t.above.ll
// RUN: ondrix-compile %S/Inputs/ox_library_above.ox --emit=c-header > %t.above.h
// RUN: llc -relocation-model=pic -filetype=obj %t.above.ll -o %t.above.o
// RUN: cc -std=gnu11 -O2 -ffp-contract=off -include %t.blend.h -include %t.spread.h -include %t.above.h %S/Inputs/ox_library_values_aot.c %t.blend.o %t.spread.o %t.above.o -lm -o %t
// RUN: %t
// RUN: opt -O3 -S %t.blend.ll -o %t.blend.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.blend.opt.ll -o %t.blend.opt.o
// RUN: opt -O3 -S %t.spread.ll -o %t.spread.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.spread.opt.ll -o %t.spread.opt.o
// RUN: opt -O3 -S %t.above.ll -o %t.above.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.above.opt.ll -o %t.above.opt.o
// RUN: cc -std=gnu11 -O2 -ffp-contract=off -include %t.blend.h -include %t.spread.h -include %t.above.h %S/Inputs/ox_library_values_aot.c %t.blend.opt.o %t.spread.opt.o %t.above.opt.o -lm -o %t.opt
// RUN: %t.opt

// Reductions inside ordinary expressions against C references at LLVM O0 and O3:
// a loop-carried dot, a callee, rms and a repeated operand, and a bool body.

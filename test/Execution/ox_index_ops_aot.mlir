// RUN: ondrix-compile %S/Inputs/ox_ix_decimate.ox --emit=llvmir --llvm-opt-level=0 > %t.decimate.ll
// RUN: ondrix-compile %S/Inputs/ox_ix_decimate.ox --emit=c-header > %t.decimate.h
// RUN: llc -relocation-model=pic -filetype=obj %t.decimate.ll -o %t.decimate.o
// RUN: ondrix-compile %S/Inputs/ox_ix_hold.ox --emit=llvmir --llvm-opt-level=0 > %t.hold.ll
// RUN: ondrix-compile %S/Inputs/ox_ix_hold.ox --emit=c-header > %t.hold.h
// RUN: llc -relocation-model=pic -filetype=obj %t.hold.ll -o %t.hold.o
// RUN: ondrix-compile %S/Inputs/ox_ix_traps.ox --emit=llvmir --llvm-opt-level=0 > %t.traps.ll
// RUN: ondrix-compile %S/Inputs/ox_ix_traps.ox --emit=c-header > %t.traps.h
// RUN: llc -relocation-model=pic -filetype=obj %t.traps.ll -o %t.traps.o
// RUN: cc -std=gnu11 -O2 -ffp-contract=off -include %t.decimate.h -include %t.hold.h -include %t.traps.h %S/Inputs/ox_index_ops_aot.c %t.decimate.o %t.hold.o %t.traps.o -o %t
// RUN: %t
// RUN: opt -O3 -S %t.decimate.ll -o %t.decimate.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.decimate.opt.ll -o %t.decimate.opt.o
// RUN: opt -O3 -S %t.hold.ll -o %t.hold.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.hold.opt.ll -o %t.hold.opt.o
// RUN: opt -O3 -S %t.traps.ll -o %t.traps.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.traps.opt.ll -o %t.traps.opt.o
// RUN: cc -std=gnu11 -O2 -ffp-contract=off -include %t.decimate.h -include %t.hold.h -include %t.traps.h %S/Inputs/ox_index_ops_aot.c %t.decimate.opt.o %t.hold.opt.o %t.traps.opt.o -o %t.opt
// RUN: %t.opt

// Index * and / against C references at LLVM O0 and O3: strided and truncated subscripts,
// and products or quotients that leave the 64-bit index range next to ones that fit.

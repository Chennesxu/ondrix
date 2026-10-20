// RUN: ondrix-compile %S/Inputs/ox_loop_energy.ox --emit=llvmir --llvm-opt-level=0 > %t.energy.ll
// RUN: ondrix-compile %S/Inputs/ox_loop_energy.ox --emit=c-header > %t.energy.h
// RUN: llc -relocation-model=pic -filetype=obj %t.energy.ll -o %t.energy.o
// RUN: ondrix-compile %S/Inputs/ox_loop_mix.ox --emit=llvmir --llvm-opt-level=0 > %t.mix.ll
// RUN: ondrix-compile %S/Inputs/ox_loop_mix.ox --emit=c-header > %t.mix.h
// RUN: llc -relocation-model=pic -filetype=obj %t.mix.ll -o %t.mix.o
// RUN: ondrix-compile %S/Inputs/ox_loop_tensor.ox --emit=llvmir --llvm-opt-level=0 > %t.tensor.ll
// RUN: ondrix-compile %S/Inputs/ox_loop_tensor.ox --emit=c-header > %t.tensor.h
// RUN: llc -relocation-model=pic -filetype=obj %t.tensor.ll -o %t.tensor.o
// RUN: ondrix-compile %S/Inputs/ox_loop_i32.ox --emit=llvmir --llvm-opt-level=0 > %t.i32.ll
// RUN: ondrix-compile %S/Inputs/ox_loop_i32.ox --emit=c-header > %t.i32.h
// RUN: llc -relocation-model=pic -filetype=obj %t.i32.ll -o %t.i32.o
// RUN: ondrix-compile %S/Inputs/ox_loop_i64.ox --emit=llvmir --llvm-opt-level=0 > %t.i64.ll
// RUN: ondrix-compile %S/Inputs/ox_loop_i64.ox --emit=c-header > %t.i64.h
// RUN: llc -relocation-model=pic -filetype=obj %t.i64.ll -o %t.i64.o
// RUN: ondrix-compile %S/Inputs/ox_loop_at.ox --emit=llvmir --llvm-opt-level=0 > %t.at.ll
// RUN: ondrix-compile %S/Inputs/ox_loop_at.ox --emit=c-header > %t.at.h
// RUN: llc -relocation-model=pic -filetype=obj %t.at.ll -o %t.at.o
// RUN: ondrix-compile %S/Inputs/ox_loop_overrun.ox --emit=llvmir --llvm-opt-level=0 > %t.overrun.ll
// RUN: ondrix-compile %S/Inputs/ox_loop_overrun.ox --emit=c-header > %t.overrun.h
// RUN: llc -relocation-model=pic -filetype=obj %t.overrun.ll -o %t.overrun.o
// RUN: cc -std=gnu11 -O2 -ffp-contract=off -include %t.energy.h -include %t.mix.h -include %t.tensor.h -include %t.i32.h -include %t.i64.h -include %t.at.h -include %t.overrun.h %S/Inputs/ox_loops_aot.c %t.energy.o %t.mix.o %t.tensor.o %t.i32.o %t.i64.o %t.at.o %t.overrun.o -o %t
// RUN: %t
// RUN: opt -O3 -S %t.energy.ll -o %t.energy.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.energy.opt.ll -o %t.energy.opt.o
// RUN: opt -O3 -S %t.mix.ll -o %t.mix.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.mix.opt.ll -o %t.mix.opt.o
// RUN: opt -O3 -S %t.tensor.ll -o %t.tensor.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.tensor.opt.ll -o %t.tensor.opt.o
// RUN: opt -O3 -S %t.i32.ll -o %t.i32.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.i32.opt.ll -o %t.i32.opt.o
// RUN: opt -O3 -S %t.i64.ll -o %t.i64.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.i64.opt.ll -o %t.i64.opt.o
// RUN: opt -O3 -S %t.at.ll -o %t.at.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.at.opt.ll -o %t.at.opt.o
// RUN: opt -O3 -S %t.overrun.ll -o %t.overrun.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.overrun.opt.ll -o %t.overrun.opt.o
// RUN: cc -std=gnu11 -O2 -ffp-contract=off -include %t.energy.h -include %t.mix.h -include %t.tensor.h -include %t.i32.h -include %t.i64.h -include %t.at.h -include %t.overrun.h %S/Inputs/ox_loops_aot.c %t.energy.opt.o %t.mix.opt.o %t.tensor.opt.o %t.i32.opt.o %t.i64.opt.o %t.at.opt.o %t.overrun.opt.o -o %t.opt
// RUN: %t.opt

// Source loops against independent C references at LLVM O0 and O3: every range
// form at the i32/i64 limits, order-sensitive f32 recurrences, and subscript traps.

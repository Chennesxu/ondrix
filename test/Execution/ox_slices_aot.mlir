// RUN: ondrix-compile %S/Inputs/ox_sl_qfir.ox --emit=llvmir --llvm-opt-level=0 > %t.qfir.ll
// RUN: ondrix-compile %S/Inputs/ox_sl_qfir.ox --emit=c-header > %t.qfir.h
// RUN: llc -relocation-model=pic -filetype=obj %t.qfir.ll -o %t.qfir.o
// RUN: ondrix-compile %S/Inputs/ox_sl_win.ox --emit=llvmir --llvm-opt-level=0 > %t.win.ll
// RUN: ondrix-compile %S/Inputs/ox_sl_win.ox --emit=c-header > %t.win.h
// RUN: llc -relocation-model=pic -filetype=obj %t.win.ll -o %t.win.o
// RUN: ondrix-compile %S/Inputs/ox_sl_scratch.ox --emit=llvmir --llvm-opt-level=0 > %t.scratch.ll
// RUN: ondrix-compile %S/Inputs/ox_sl_scratch.ox --emit=c-header > %t.scratch.h
// RUN: llc -relocation-model=pic -filetype=obj %t.scratch.ll -o %t.scratch.o
// RUN: ondrix-compile %S/Inputs/ox_sl_taps.ox --emit=llvmir --llvm-opt-level=0 > %t.taps.ll
// RUN: ondrix-compile %S/Inputs/ox_sl_taps.ox --emit=c-header > %t.taps.h
// RUN: llc -relocation-model=pic -filetype=obj %t.taps.ll -o %t.taps.o
// RUN: ondrix-compile %S/Inputs/ox_sl_delay.ox --emit=llvmir --llvm-opt-level=0 > %t.delay.ll
// RUN: ondrix-compile %S/Inputs/ox_sl_delay.ox --emit=c-header > %t.delay.h
// RUN: llc -relocation-model=pic -filetype=obj %t.delay.ll -o %t.delay.o
// RUN: ondrix-compile %S/Inputs/ox_sl_trap.ox --emit=llvmir --llvm-opt-level=0 > %t.trap.ll
// RUN: ondrix-compile %S/Inputs/ox_sl_trap.ox --emit=c-header > %t.trap.h
// RUN: llc -relocation-model=pic -filetype=obj %t.trap.ll -o %t.trap.o
// RUN: ondrix-compile %S/Inputs/ox_sl_tail.ox --emit=llvmir --llvm-opt-level=0 > %t.tail.ll
// RUN: ondrix-compile %S/Inputs/ox_sl_tail.ox --emit=c-header > %t.tail.h
// RUN: llc -relocation-model=pic -filetype=obj %t.tail.ll -o %t.tail.o
// RUN: cc -std=gnu11 -O2 -ffp-contract=off -include %t.qfir.h -include %t.win.h -include %t.scratch.h -include %t.taps.h -include %t.delay.h -include %t.trap.h -include %t.tail.h %S/Inputs/ox_slices_aot.c %t.qfir.o %t.win.o %t.scratch.o %t.taps.o %t.delay.o %t.trap.o %t.tail.o -lm -o %t
// RUN: %t
// RUN: opt -O3 -S %t.qfir.ll -o %t.qfir.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.qfir.opt.ll -o %t.qfir.opt.o
// RUN: opt -O3 -S %t.win.ll -o %t.win.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.win.opt.ll -o %t.win.opt.o
// RUN: opt -O3 -S %t.scratch.ll -o %t.scratch.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.scratch.opt.ll -o %t.scratch.opt.o
// RUN: opt -O3 -S %t.taps.ll -o %t.taps.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.taps.opt.ll -o %t.taps.opt.o
// RUN: opt -O3 -S %t.delay.ll -o %t.delay.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.delay.opt.ll -o %t.delay.opt.o
// RUN: opt -O3 -S %t.trap.ll -o %t.trap.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.trap.opt.ll -o %t.trap.opt.o
// RUN: opt -O3 -S %t.tail.ll -o %t.tail.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.tail.opt.ll -o %t.tail.opt.o
// RUN: cc -std=gnu11 -O2 -ffp-contract=off -include %t.qfir.h -include %t.win.h -include %t.scratch.h -include %t.taps.h -include %t.delay.h -include %t.trap.h -include %t.tail.h %S/Inputs/ox_slices_aot.c %t.qfir.opt.o %t.win.opt.o %t.scratch.opt.o %t.taps.opt.o %t.delay.opt.o %t.trap.opt.o %t.tail.opt.o -lm -o %t.opt
// RUN: %t.opt

// Slices and local arrays against C references at LLVM O0 and O3: sliding q15 and f32 windows,
// a static rms slice, a tensor tail, re-zeroed scratch, a view of a mut buffer, empty views, traps.

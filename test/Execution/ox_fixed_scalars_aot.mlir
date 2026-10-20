// RUN: ondrix-compile %S/Inputs/ox_fixed_mac.ox --emit=llvmir --llvm-opt-level=0 > %t.mac.ll
// RUN: ondrix-compile %S/Inputs/ox_fixed_mac.ox --emit=c-header > %t.mac.h
// RUN: llc -relocation-model=pic -filetype=obj %t.mac.ll -o %t.mac.o
// RUN: ondrix-compile %S/Inputs/ox_fixed_pole.ox --emit=llvmir --llvm-opt-level=0 > %t.pole.ll
// RUN: ondrix-compile %S/Inputs/ox_fixed_pole.ox --emit=c-header > %t.pole.h
// RUN: llc -relocation-model=pic -filetype=obj %t.pole.ll -o %t.pole.o
// RUN: ondrix-compile %S/Inputs/ox_fixed_gap.ox --emit=llvmir --llvm-opt-level=0 > %t.gap.ll
// RUN: ondrix-compile %S/Inputs/ox_fixed_gap.ox --emit=c-header > %t.gap.h
// RUN: llc -relocation-model=pic -filetype=obj %t.gap.ll -o %t.gap.o
// RUN: ondrix-compile %S/Inputs/ox_fixed_mean.ox --emit=llvmir --llvm-opt-level=0 > %t.mean.ll
// RUN: ondrix-compile %S/Inputs/ox_fixed_mean.ox --emit=c-header > %t.mean.h
// RUN: llc -relocation-model=pic -filetype=obj %t.mean.ll -o %t.mean.o
// RUN: ondrix-compile %S/Inputs/ox_fixed_above.ox --emit=llvmir --llvm-opt-level=0 > %t.above.ll
// RUN: ondrix-compile %S/Inputs/ox_fixed_above.ox --emit=c-header > %t.above.h
// RUN: llc -relocation-model=pic -filetype=obj %t.above.ll -o %t.above.o
// RUN: cc -std=gnu11 -O2 -ffp-contract=off -include %t.mac.h -include %t.pole.h -include %t.gap.h -include %t.mean.h -include %t.above.h %S/Inputs/ox_fixed_scalars_aot.c %t.mac.o %t.pole.o %t.gap.o %t.mean.o %t.above.o -o %t
// RUN: %t
// RUN: opt -O3 -S %t.mac.ll -o %t.mac.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.mac.opt.ll -o %t.mac.opt.o
// RUN: opt -O3 -S %t.pole.ll -o %t.pole.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.pole.opt.ll -o %t.pole.opt.o
// RUN: opt -O3 -S %t.gap.ll -o %t.gap.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.gap.opt.ll -o %t.gap.opt.o
// RUN: opt -O3 -S %t.mean.ll -o %t.mean.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.mean.opt.ll -o %t.mean.opt.o
// RUN: opt -O3 -S %t.above.ll -o %t.above.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.above.opt.ll -o %t.above.opt.o
// RUN: cc -std=gnu11 -O2 -ffp-contract=off -include %t.mac.h -include %t.pole.h -include %t.gap.h -include %t.mean.h -include %t.above.h %S/Inputs/ox_fixed_scalars_aot.c %t.mac.opt.o %t.pole.opt.o %t.gap.opt.o %t.mean.opt.o %t.above.opt.o -o %t.opt
// RUN: %t.opt

// Fixed-point scalar programs against C references at LLVM O0 and O3: a Q15 MAC
// through q31, a rounded one-pole recurrence, a dot value, a dequantized sum.

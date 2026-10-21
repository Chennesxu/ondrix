// RUN: ondrix-compile %S/Inputs/ox_acc_mac.ox --emit=llvmir --llvm-opt-level=0 > %t.mac.ll
// RUN: ondrix-compile %S/Inputs/ox_acc_mac.ox --emit=c-header > %t.mac.h
// RUN: llc -relocation-model=pic -filetype=obj %t.mac.ll -o %t.mac.o
// RUN: ondrix-compile %S/Inputs/ox_acc_dot.ox --emit=llvmir --llvm-opt-level=0 > %t.dot.ll
// RUN: ondrix-compile %S/Inputs/ox_acc_dot.ox --emit=c-header > %t.dot.h
// RUN: llc -relocation-model=pic -filetype=obj %t.dot.ll -o %t.dot.o
// RUN: ondrix-compile %S/Inputs/ox_acc_wrap.ox --emit=llvmir --llvm-opt-level=0 > %t.wrap.ll
// RUN: ondrix-compile %S/Inputs/ox_acc_wrap.ox --emit=c-header > %t.wrap.h
// RUN: llc -relocation-model=pic -filetype=obj %t.wrap.ll -o %t.wrap.o
// RUN: ondrix-compile %S/Inputs/ox_acc_dot_wrap.ox --emit=llvmir --llvm-opt-level=0 > %t.dot_wrap.ll
// RUN: ondrix-compile %S/Inputs/ox_acc_dot_wrap.ox --emit=c-header > %t.dot_wrap.h
// RUN: llc -relocation-model=pic -filetype=obj %t.dot_wrap.ll -o %t.dot_wrap.o
// RUN: ondrix-compile %S/Inputs/ox_acc_pole.ox --emit=llvmir --llvm-opt-level=0 > %t.pole.ll
// RUN: ondrix-compile %S/Inputs/ox_acc_pole.ox --emit=c-header > %t.pole.h
// RUN: llc -relocation-model=pic -filetype=obj %t.pole.ll -o %t.pole.o
// RUN: ondrix-compile %S/Inputs/ox_acc_leak.ox --emit=llvmir --llvm-opt-level=0 > %t.leak.ll
// RUN: ondrix-compile %S/Inputs/ox_acc_leak.ox --emit=c-header > %t.leak.h
// RUN: llc -relocation-model=pic -filetype=obj %t.leak.ll -o %t.leak.o
// RUN: ondrix-compile %S/Inputs/ox_acc_q31.ox --emit=llvmir --llvm-opt-level=0 > %t.q31.ll
// RUN: ondrix-compile %S/Inputs/ox_acc_q31.ox --emit=c-header > %t.q31.h
// RUN: llc -relocation-model=pic -filetype=obj %t.q31.ll -o %t.q31.o
// RUN: cc -std=gnu11 -O2 -include %t.mac.h -include %t.dot.h -include %t.wrap.h -include %t.dot_wrap.h -include %t.pole.h -include %t.leak.h -include %t.q31.h %S/Inputs/ox_accumulators_aot.c %t.mac.o %t.dot.o %t.wrap.o %t.dot_wrap.o %t.pole.o %t.leak.o %t.q31.o -o %t
// RUN: %t
// RUN: opt -O3 -S %t.mac.ll -o %t.mac.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.mac.opt.ll -o %t.mac.opt.o
// RUN: opt -O3 -S %t.dot.ll -o %t.dot.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.dot.opt.ll -o %t.dot.opt.o
// RUN: opt -O3 -S %t.wrap.ll -o %t.wrap.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.wrap.opt.ll -o %t.wrap.opt.o
// RUN: opt -O3 -S %t.dot_wrap.ll -o %t.dot_wrap.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.dot_wrap.opt.ll -o %t.dot_wrap.opt.o
// RUN: opt -O3 -S %t.pole.ll -o %t.pole.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.pole.opt.ll -o %t.pole.opt.o
// RUN: opt -O3 -S %t.leak.ll -o %t.leak.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.leak.opt.ll -o %t.leak.opt.o
// RUN: opt -O3 -S %t.q31.ll -o %t.q31.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.q31.opt.ll -o %t.q31.opt.o
// RUN: cc -std=gnu11 -O2 -include %t.mac.h -include %t.dot.h -include %t.wrap.h -include %t.dot_wrap.h -include %t.pole.h -include %t.leak.h -include %t.q31.h %S/Inputs/ox_accumulators_aot.c %t.mac.opt.o %t.dot.opt.o %t.wrap.opt.o %t.dot_wrap.opt.o %t.pole.opt.o %t.leak.opt.o %t.q31.opt.o -o %t.opt
// RUN: %t.opt

// Source accumulators against C references at LLVM O0 and O3: a loop MAC equal to dot
// under saturate and wrap, a pole rounding once per sample, import with mac_sub, Q31.

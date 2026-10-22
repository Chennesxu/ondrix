// RUN: ondrix-compile %S/Inputs/ox_cv_mean.ox --emit=llvmir --llvm-opt-level=0 > %t.mean.ll
// RUN: ondrix-compile %S/Inputs/ox_cv_mean.ox --emit=c-header > %t.mean.h
// RUN: llc -relocation-model=pic -filetype=obj %t.mean.ll -o %t.mean.o
// RUN: ondrix-compile %S/Inputs/ox_cv_round.ox --emit=llvmir --llvm-opt-level=0 > %t.round.ll
// RUN: ondrix-compile %S/Inputs/ox_cv_round.ox --emit=c-header > %t.round.h
// RUN: llc -relocation-model=pic -filetype=obj %t.round.ll -o %t.round.o
// RUN: ondrix-compile %S/Inputs/ox_cv_round64.ox --emit=llvmir --llvm-opt-level=0 > %t.round64.ll
// RUN: ondrix-compile %S/Inputs/ox_cv_round64.ox --emit=c-header > %t.round64.h
// RUN: llc -relocation-model=pic -filetype=obj %t.round64.ll -o %t.round64.o
// RUN: ondrix-compile %S/Inputs/ox_cv_narrow.ox --emit=llvmir --llvm-opt-level=0 > %t.narrow.ll
// RUN: ondrix-compile %S/Inputs/ox_cv_narrow.ox --emit=c-header > %t.narrow.h
// RUN: llc -relocation-model=pic -filetype=obj %t.narrow.ll -o %t.narrow.o
// RUN: ondrix-compile %S/Inputs/ox_cv_widen.ox --emit=llvmir --llvm-opt-level=0 > %t.widen.ll
// RUN: ondrix-compile %S/Inputs/ox_cv_widen.ox --emit=c-header > %t.widen.h
// RUN: llc -relocation-model=pic -filetype=obj %t.widen.ll -o %t.widen.o
// RUN: ondrix-compile %S/Inputs/ox_cv_float.ox --emit=llvmir --llvm-opt-level=0 > %t.float.ll
// RUN: ondrix-compile %S/Inputs/ox_cv_float.ox --emit=c-header > %t.float.h
// RUN: llc -relocation-model=pic -filetype=obj %t.float.ll -o %t.float.o
// RUN: ondrix-compile %S/Inputs/ox_cv_tail.ox --emit=llvmir --llvm-opt-level=0 > %t.tail.ll
// RUN: ondrix-compile %S/Inputs/ox_cv_tail.ox --emit=c-header > %t.tail.h
// RUN: llc -relocation-model=pic -filetype=obj %t.tail.ll -o %t.tail.o
// RUN: cc -std=gnu11 -O2 -ffp-contract=off -include %t.mean.h -include %t.round.h -include %t.round64.h -include %t.narrow.h -include %t.widen.h -include %t.float.h -include %t.tail.h %S/Inputs/ox_convert_aot.c %t.mean.o %t.round.o %t.round64.o %t.narrow.o %t.widen.o %t.float.o %t.tail.o -lm -o %t
// RUN: %t
// RUN: opt -O3 -S %t.mean.ll -o %t.mean.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.mean.opt.ll -o %t.mean.opt.o
// RUN: opt -O3 -S %t.round.ll -o %t.round.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.round.opt.ll -o %t.round.opt.o
// RUN: opt -O3 -S %t.round64.ll -o %t.round64.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.round64.opt.ll -o %t.round64.opt.o
// RUN: opt -O3 -S %t.narrow.ll -o %t.narrow.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.narrow.opt.ll -o %t.narrow.opt.o
// RUN: opt -O3 -S %t.widen.ll -o %t.widen.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.widen.opt.ll -o %t.widen.opt.o
// RUN: opt -O3 -S %t.float.ll -o %t.float.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.float.opt.ll -o %t.float.opt.o
// RUN: opt -O3 -S %t.tail.ll -o %t.tail.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.tail.opt.ll -o %t.tail.opt.o
// RUN: cc -std=gnu11 -O2 -ffp-contract=off -include %t.mean.h -include %t.round.h -include %t.round64.h -include %t.narrow.h -include %t.widen.h -include %t.float.h -include %t.tail.h %S/Inputs/ox_convert_aot.c %t.mean.opt.o %t.round.opt.o %t.round64.opt.o %t.narrow.opt.o %t.widen.opt.o %t.float.opt.o %t.tail.opt.o -lm -o %t.opt
// RUN: %t.opt

// convert against C references at LLVM O0 and O3: the four rounding modes at both
// widths, both narrowing policies, sign-extending integer and index conversions.

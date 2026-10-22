// RUN: ondrix-compile %S/Inputs/ox_it_hist.ox --emit=llvmir --llvm-opt-level=0 > %t.hist.ll
// RUN: ondrix-compile %S/Inputs/ox_it_hist.ox --emit=c-header > %t.hist.h
// RUN: llc -relocation-model=pic -filetype=obj %t.hist.ll -o %t.hist.o
// RUN: ondrix-compile %S/Inputs/ox_it_gather.ox --emit=llvmir --llvm-opt-level=0 > %t.gather.ll
// RUN: ondrix-compile %S/Inputs/ox_it_gather.ox --emit=c-header > %t.gather.h
// RUN: llc -relocation-model=pic -filetype=obj %t.gather.ll -o %t.gather.o
// RUN: ondrix-compile %S/Inputs/ox_it_pick.ox --emit=llvmir --llvm-opt-level=0 > %t.pick.ll
// RUN: ondrix-compile %S/Inputs/ox_it_pick.ox --emit=c-header > %t.pick.h
// RUN: llc -relocation-model=pic -filetype=obj %t.pick.ll -o %t.pick.o
// RUN: ondrix-compile %S/Inputs/ox_it_wrap.ox --emit=llvmir --llvm-opt-level=0 > %t.wrap.ll
// RUN: ondrix-compile %S/Inputs/ox_it_wrap.ox --emit=c-header > %t.wrap.h
// RUN: llc -relocation-model=pic -filetype=obj %t.wrap.ll -o %t.wrap.o
// RUN: ondrix-compile %S/Inputs/ox_it_div.ox --emit=llvmir --llvm-opt-level=0 > %t.div.ll
// RUN: ondrix-compile %S/Inputs/ox_it_div.ox --emit=c-header > %t.div.h
// RUN: llc -relocation-model=pic -filetype=obj %t.div.ll -o %t.div.o
// RUN: ondrix-compile %S/Inputs/ox_it_quant.ox --emit=llvmir --llvm-opt-level=0 > %t.quant.ll
// RUN: ondrix-compile %S/Inputs/ox_it_quant.ox --emit=c-header > %t.quant.h
// RUN: llc -relocation-model=pic -filetype=obj %t.quant.ll -o %t.quant.o
// RUN: ondrix-compile %S/Inputs/ox_it_narrow.ox --emit=llvmir --llvm-opt-level=0 > %t.narrow.ll
// RUN: ondrix-compile %S/Inputs/ox_it_narrow.ox --emit=c-header > %t.narrow.h
// RUN: llc -relocation-model=pic -filetype=obj %t.narrow.ll -o %t.narrow.o
// RUN: ondrix-compile %S/Inputs/ox_it_widen.ox --emit=llvmir --llvm-opt-level=0 > %t.widen.ll
// RUN: ondrix-compile %S/Inputs/ox_it_widen.ox --emit=c-header > %t.widen.h
// RUN: llc -relocation-model=pic -filetype=obj %t.widen.ll -o %t.widen.o
// RUN: ondrix-compile %S/Inputs/ox_it_local.ox --emit=llvmir --llvm-opt-level=0 > %t.local.ll
// RUN: ondrix-compile %S/Inputs/ox_it_local.ox --emit=c-header > %t.local.h
// RUN: llc -relocation-model=pic -filetype=obj %t.local.ll -o %t.local.o
// RUN: cc -std=gnu11 -O2 -ffp-contract=off -include %t.hist.h -include %t.gather.h -include %t.pick.h -include %t.wrap.h -include %t.div.h -include %t.quant.h -include %t.narrow.h -include %t.widen.h -include %t.local.h %S/Inputs/ox_int_tensors_aot.c %t.hist.o %t.gather.o %t.pick.o %t.wrap.o %t.div.o %t.quant.o %t.narrow.o %t.widen.o %t.local.o -lm -o %t
// RUN: %t
// RUN: opt -O3 -S %t.hist.ll -o %t.hist.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.hist.opt.ll -o %t.hist.opt.o
// RUN: opt -O3 -S %t.gather.ll -o %t.gather.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.gather.opt.ll -o %t.gather.opt.o
// RUN: opt -O3 -S %t.pick.ll -o %t.pick.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.pick.opt.ll -o %t.pick.opt.o
// RUN: opt -O3 -S %t.wrap.ll -o %t.wrap.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.wrap.opt.ll -o %t.wrap.opt.o
// RUN: opt -O3 -S %t.div.ll -o %t.div.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.div.opt.ll -o %t.div.opt.o
// RUN: opt -O3 -S %t.quant.ll -o %t.quant.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.quant.opt.ll -o %t.quant.opt.o
// RUN: opt -O3 -S %t.narrow.ll -o %t.narrow.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.narrow.opt.ll -o %t.narrow.opt.o
// RUN: opt -O3 -S %t.widen.ll -o %t.widen.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.widen.opt.ll -o %t.widen.opt.o
// RUN: opt -O3 -S %t.local.ll -o %t.local.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.local.opt.ll -o %t.local.opt.o
// RUN: cc -std=gnu11 -O2 -ffp-contract=off -include %t.hist.h -include %t.gather.h -include %t.pick.h -include %t.wrap.h -include %t.div.h -include %t.quant.h -include %t.narrow.h -include %t.widen.h -include %t.local.h %S/Inputs/ox_int_tensors_aot.c %t.hist.opt.o %t.gather.opt.o %t.pick.opt.o %t.wrap.opt.o %t.div.opt.o %t.quant.opt.o %t.narrow.opt.o %t.widen.opt.o %t.local.opt.o -lm -o %t.opt
// RUN: %t.opt

// i32 and i64 buffers, arrays, slices and tensors against C references at LLVM O0
// and O3: wrapping and checked tensor arithmetic, per-element convert, subscript traps.

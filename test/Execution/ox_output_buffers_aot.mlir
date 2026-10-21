// RUN: ondrix-compile %S/Inputs/ox_out_scale.ox --emit=llvmir --llvm-opt-level=0 > %t.scale.ll
// RUN: ondrix-compile %S/Inputs/ox_out_scale.ox --emit=c-header > %t.scale.h
// RUN: llc -relocation-model=pic -filetype=obj %t.scale.ll -o %t.scale.o
// RUN: ondrix-compile %S/Inputs/ox_out_fir.ox --emit=llvmir --llvm-opt-level=0 > %t.fir.ll
// RUN: ondrix-compile %S/Inputs/ox_out_fir.ox --emit=c-header > %t.fir.h
// RUN: llc -relocation-model=pic -filetype=obj %t.fir.ll -o %t.fir.o
// RUN: ondrix-compile %S/Inputs/ox_out_delay.ox --emit=llvmir --llvm-opt-level=0 > %t.delay.ll
// RUN: ondrix-compile %S/Inputs/ox_out_delay.ox --emit=c-header > %t.delay.h
// RUN: llc -relocation-model=pic -filetype=obj %t.delay.ll -o %t.delay.o
// RUN: ondrix-compile %S/Inputs/ox_out_prefix.ox --emit=llvmir --llvm-opt-level=0 > %t.prefix.ll
// RUN: ondrix-compile %S/Inputs/ox_out_prefix.ox --emit=c-header > %t.prefix.h
// RUN: llc -relocation-model=pic -filetype=obj %t.prefix.ll -o %t.prefix.o
// RUN: ondrix-compile %S/Inputs/ox_out_put.ox --emit=llvmir --llvm-opt-level=0 > %t.put.ll
// RUN: ondrix-compile %S/Inputs/ox_out_put.ox --emit=c-header > %t.put.h
// RUN: llc -relocation-model=pic -filetype=obj %t.put.ll -o %t.put.o
// RUN: ondrix-compile %S/Inputs/ox_out_order.ox --emit=llvmir --llvm-opt-level=0 > %t.order.ll
// RUN: ondrix-compile %S/Inputs/ox_out_order.ox --emit=c-header > %t.order.h
// RUN: llc -relocation-model=pic -filetype=obj %t.order.ll -o %t.order.o
// RUN: cc -std=gnu11 -O2 -include %t.scale.h -include %t.fir.h -include %t.delay.h -include %t.prefix.h -include %t.put.h -include %t.order.h %S/Inputs/ox_output_buffers_aot.c %t.scale.o %t.fir.o %t.delay.o %t.prefix.o %t.put.o %t.order.o -o %t
// RUN: %t
// RUN: opt -O3 -S %t.scale.ll -o %t.scale.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.scale.opt.ll -o %t.scale.opt.o
// RUN: opt -O3 -S %t.fir.ll -o %t.fir.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.fir.opt.ll -o %t.fir.opt.o
// RUN: opt -O3 -S %t.delay.ll -o %t.delay.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.delay.opt.ll -o %t.delay.opt.o
// RUN: opt -O3 -S %t.prefix.ll -o %t.prefix.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.prefix.opt.ll -o %t.prefix.opt.o
// RUN: opt -O3 -S %t.put.ll -o %t.put.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.put.opt.ll -o %t.put.opt.o
// RUN: opt -O3 -S %t.order.ll -o %t.order.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.order.opt.ll -o %t.order.opt.o
// RUN: cc -std=gnu11 -O2 -include %t.scale.h -include %t.fir.h -include %t.delay.h -include %t.prefix.h -include %t.put.h -include %t.order.h %S/Inputs/ox_output_buffers_aot.c %t.scale.opt.o %t.fir.opt.o %t.delay.opt.o %t.prefix.opt.o %t.put.opt.o %t.order.opt.o -o %t.opt
// RUN: %t.opt
// RUN: ondrix-compile %S/Inputs/ox_out_scale.ox --checked-entries --emit=llvmir > %t.scale.checked.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.scale.checked.ll -o %t.scale.checked.o
// RUN: ondrix-compile %S/Inputs/ox_out_delay.ox --checked-entries --emit=llvmir > %t.delay.checked.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.delay.checked.ll -o %t.delay.checked.o
// RUN: cc -std=gnu11 -O2 -DCHECKED -include %t.scale.h -include %t.delay.h %S/Inputs/ox_output_buffers_aot.c %t.scale.checked.o %t.delay.checked.o -o %t.checked
// RUN: %t.checked

// Output buffers against C references at LLVM O0 and O3: scale, per-output FIR, in-place
// prefix sum and delay line, write traps and their order, checked-entry overlap refusal.

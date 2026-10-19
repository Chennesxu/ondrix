// RUN: ondrix-compile %S/Inputs/ox_basic_i32.ox --emit=llvmir --llvm-opt-level=0 > %t.i32.ll
// RUN: ondrix-compile %S/Inputs/ox_basic_i32.ox --emit=c-header > %t.i32.h
// RUN: llc -relocation-model=pic -filetype=obj %t.i32.ll -o %t.i32.o
// RUN: ondrix-compile %S/Inputs/ox_basic_i64.ox --emit=llvmir --llvm-opt-level=0 > %t.i64.ll
// RUN: ondrix-compile %S/Inputs/ox_basic_i64.ox --emit=c-header > %t.i64.h
// RUN: llc -relocation-model=pic -filetype=obj %t.i64.ll -o %t.i64.o
// RUN: ondrix-compile %S/Inputs/ox_basic_bool.ox --emit=llvmir --llvm-opt-level=0 > %t.bool.ll
// RUN: ondrix-compile %S/Inputs/ox_basic_bool.ox --emit=c-header > %t.bool.h
// RUN: llc -relocation-model=pic -filetype=obj %t.bool.ll -o %t.bool.o
// RUN: ondrix-compile %S/Inputs/ox_basic_eager.ox --emit=llvmir --llvm-opt-level=0 > %t.eager.ll
// RUN: ondrix-compile %S/Inputs/ox_basic_eager.ox --emit=c-header > %t.eager.h
// RUN: llc -relocation-model=pic -filetype=obj %t.eager.ll -o %t.eager.o
// RUN: ondrix-compile %S/Inputs/ox_basic_f32.ox --emit=llvmir --llvm-opt-level=0 > %t.f32.ll
// RUN: ondrix-compile %S/Inputs/ox_basic_f32.ox --emit=c-header > %t.f32.h
// RUN: llc -relocation-model=pic -filetype=obj %t.f32.ll -o %t.f32.o
// RUN: ondrix-compile %S/Inputs/ox_basic_f32_neg.ox --emit=llvmir --llvm-opt-level=0 > %t.f32_neg.ll
// RUN: ondrix-compile %S/Inputs/ox_basic_f32_neg.ox --emit=c-header > %t.f32_neg.h
// RUN: llc -relocation-model=pic -filetype=obj %t.f32_neg.ll -o %t.f32_neg.o
// RUN: ondrix-compile %S/Inputs/ox_basic_f32_compare.ox --emit=llvmir --llvm-opt-level=0 > %t.f32_compare.ll
// RUN: ondrix-compile %S/Inputs/ox_basic_f32_compare.ox --emit=c-header > %t.f32_compare.h
// RUN: llc -relocation-model=pic -filetype=obj %t.f32_compare.ll -o %t.f32_compare.o
// RUN: ondrix-compile %S/Inputs/ox_basic_f32_tensor.ox --emit=llvmir --llvm-opt-level=0 > %t.f32_tensor.ll
// RUN: ondrix-compile %S/Inputs/ox_basic_f32_tensor.ox --emit=c-header > %t.f32_tensor.h
// RUN: llc -relocation-model=pic -filetype=obj %t.f32_tensor.ll -o %t.f32_tensor.o
// RUN: ondrix-compile %S/Inputs/ox_basic_q15_neg.ox --emit=llvmir --llvm-opt-level=0 > %t.q15_neg.ll
// RUN: ondrix-compile %S/Inputs/ox_basic_q15_neg.ox --emit=c-header > %t.q15_neg.h
// RUN: llc -relocation-model=pic -filetype=obj %t.q15_neg.ll -o %t.q15_neg.o
// RUN: ondrix-compile %S/Inputs/ox_basic_q31_neg.ox --emit=llvmir --llvm-opt-level=0 > %t.q31_neg.ll
// RUN: ondrix-compile %S/Inputs/ox_basic_q31_neg.ox --emit=c-header > %t.q31_neg.h
// RUN: llc -relocation-model=pic -filetype=obj %t.q31_neg.ll -o %t.q31_neg.o
// RUN: cc -std=c11 -O2 -ffp-contract=off -include %t.i32.h -include %t.i64.h -include %t.bool.h -include %t.eager.h -include %t.f32.h -include %t.f32_neg.h -include %t.f32_compare.h -include %t.f32_tensor.h -include %t.q15_neg.h -include %t.q31_neg.h %S/Inputs/ox_basic_arithmetic_aot.c %t.i32.o %t.i64.o %t.bool.o %t.eager.o %t.f32.o %t.f32_neg.o %t.f32_compare.o %t.f32_tensor.o %t.q15_neg.o %t.q31_neg.o -lm -o %t
// RUN: %t
// RUN: opt -O3 -S %t.i32.ll -o %t.i32.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.i32.opt.ll -o %t.i32.opt.o
// RUN: opt -O3 -S %t.i64.ll -o %t.i64.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.i64.opt.ll -o %t.i64.opt.o
// RUN: opt -O3 -S %t.bool.ll -o %t.bool.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.bool.opt.ll -o %t.bool.opt.o
// RUN: opt -O3 -S %t.eager.ll -o %t.eager.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.eager.opt.ll -o %t.eager.opt.o
// RUN: opt -O3 -S %t.f32.ll -o %t.f32.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.f32.opt.ll -o %t.f32.opt.o
// RUN: opt -O3 -S %t.f32_neg.ll -o %t.f32_neg.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.f32_neg.opt.ll -o %t.f32_neg.opt.o
// RUN: opt -O3 -S %t.f32_compare.ll -o %t.f32_compare.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.f32_compare.opt.ll -o %t.f32_compare.opt.o
// RUN: opt -O3 -S %t.f32_tensor.ll -o %t.f32_tensor.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.f32_tensor.opt.ll -o %t.f32_tensor.opt.o
// RUN: opt -O3 -S %t.q15_neg.ll -o %t.q15_neg.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.q15_neg.opt.ll -o %t.q15_neg.opt.o
// RUN: opt -O3 -S %t.q31_neg.ll -o %t.q31_neg.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.q31_neg.opt.ll -o %t.q31_neg.opt.o
// RUN: cc -std=c11 -O2 -ffp-contract=off -include %t.i32.h -include %t.i64.h -include %t.bool.h -include %t.eager.h -include %t.f32.h -include %t.f32_neg.h -include %t.f32_compare.h -include %t.f32_tensor.h -include %t.q15_neg.h -include %t.q31_neg.h %S/Inputs/ox_basic_arithmetic_aot.c %t.i32.opt.o %t.i64.opt.o %t.bool.opt.o %t.eager.opt.o %t.f32.opt.o %t.f32_neg.opt.o %t.f32_compare.opt.o %t.f32_tensor.opt.o %t.q15_neg.opt.o %t.q31_neg.opt.o -lm -o %t.opt
// RUN: %t.opt

// Independent unsigned integer and explicitly rounded floating references.
// Generated headers exercise the public C ABI, including noncanonical boolean
// input bytes. Child processes distinguish short-circuit skips from eager local
// evaluation and require checked divisions to trap, also after LLVM O3.

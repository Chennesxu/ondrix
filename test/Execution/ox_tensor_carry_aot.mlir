// RUN: ondrix-compile %S/Inputs/ox_tc_smooth.ox --emit=llvmir --llvm-opt-level=0 > %t.smooth.ll
// RUN: ondrix-compile %S/Inputs/ox_tc_smooth.ox --emit=c-header > %t.smooth.h
// RUN: llc -relocation-model=pic -filetype=obj %t.smooth.ll -o %t.smooth.o
// RUN: ondrix-compile %S/Inputs/ox_tc_self.ox --emit=llvmir --llvm-opt-level=0 > %t.self.ll
// RUN: ondrix-compile %S/Inputs/ox_tc_self.ox --emit=c-header > %t.self.h
// RUN: llc -relocation-model=pic -filetype=obj %t.self.ll -o %t.self.o
// RUN: ondrix-compile %S/Inputs/ox_tc_while.ox --emit=llvmir --llvm-opt-level=0 > %t.while.ll
// RUN: ondrix-compile %S/Inputs/ox_tc_while.ox --emit=c-header > %t.while.h
// RUN: llc -relocation-model=pic -filetype=obj %t.while.ll -o %t.while.o
// RUN: ondrix-compile %S/Inputs/ox_tc_branch.ox --emit=llvmir --llvm-opt-level=0 > %t.branch.ll
// RUN: ondrix-compile %S/Inputs/ox_tc_branch.ox --emit=c-header > %t.branch.h
// RUN: llc -relocation-model=pic -filetype=obj %t.branch.ll -o %t.branch.o
// RUN: ondrix-compile %S/Inputs/ox_tc_swap.ox --emit=llvmir --llvm-opt-level=0 > %t.swap.ll
// RUN: ondrix-compile %S/Inputs/ox_tc_swap.ox --emit=c-header > %t.swap.h
// RUN: llc -relocation-model=pic -filetype=obj %t.swap.ll -o %t.swap.o
// RUN: ondrix-compile %S/Inputs/ox_tc_alias.ox --emit=llvmir --llvm-opt-level=0 > %t.alias.ll
// RUN: ondrix-compile %S/Inputs/ox_tc_alias.ox --emit=c-header > %t.alias.h
// RUN: llc -relocation-model=pic -filetype=obj %t.alias.ll -o %t.alias.o
// RUN: ondrix-compile %S/Inputs/ox_tc_nest.ox --emit=llvmir --llvm-opt-level=0 > %t.nest.ll
// RUN: ondrix-compile %S/Inputs/ox_tc_nest.ox --emit=c-header > %t.nest.h
// RUN: llc -relocation-model=pic -filetype=obj %t.nest.ll -o %t.nest.o
// RUN: ondrix-compile %S/Inputs/ox_tc_q15.ox --emit=llvmir --llvm-opt-level=0 > %t.q15.ll
// RUN: ondrix-compile %S/Inputs/ox_tc_q15.ox --emit=c-header > %t.q15.h
// RUN: llc -relocation-model=pic -filetype=obj %t.q15.ll -o %t.q15.o
// RUN: ondrix-compile %S/Inputs/ox_tc_read.ox --emit=llvmir --llvm-opt-level=0 > %t.read.ll
// RUN: ondrix-compile %S/Inputs/ox_tc_read.ox --emit=c-header > %t.read.h
// RUN: llc -relocation-model=pic -filetype=obj %t.read.ll -o %t.read.o
// RUN: ondrix-compile %S/Inputs/ox_tc_outer.ox --emit=llvmir --llvm-opt-level=0 > %t.outer.ll
// RUN: ondrix-compile %S/Inputs/ox_tc_outer.ox --emit=c-header > %t.outer.h
// RUN: llc -relocation-model=pic -filetype=obj %t.outer.ll -o %t.outer.o
// RUN: ondrix-compile %S/Inputs/ox_tc_mix.ox --emit=llvmir --llvm-opt-level=0 > %t.mix.ll
// RUN: ondrix-compile %S/Inputs/ox_tc_mix.ox --emit=c-header > %t.mix.h
// RUN: llc -relocation-model=pic -filetype=obj %t.mix.ll -o %t.mix.o
// RUN: ondrix-compile %S/Inputs/ox_tc_pick.ox --emit=llvmir --llvm-opt-level=0 > %t.pick.ll
// RUN: ondrix-compile %S/Inputs/ox_tc_named.ox --emit=llvmir --llvm-opt-level=0 > %t.named.ll
// RUN: ondrix-compile %S/Inputs/ox_tc_pick.ox --emit=c-header > %t.pick.h
// RUN: ondrix-compile %S/Inputs/ox_tc_named.ox --emit=c-header > %t.named.h
// RUN: llc -relocation-model=pic -filetype=obj %t.pick.ll -o %t.pick.o
// RUN: llc -relocation-model=pic -filetype=obj %t.named.ll -o %t.named.o
// RUN: cc -std=gnu11 -O2 -ffp-contract=off -include %t.smooth.h -include %t.self.h -include %t.while.h -include %t.branch.h -include %t.swap.h -include %t.alias.h -include %t.nest.h -include %t.q15.h -include %t.read.h -include %t.outer.h -include %t.mix.h -include %t.pick.h -include %t.named.h %S/Inputs/ox_tensor_carry_aot.c %t.smooth.o %t.self.o %t.while.o %t.branch.o %t.swap.o %t.alias.o %t.nest.o %t.q15.o %t.read.o %t.outer.o %t.mix.o %t.pick.o %t.named.o -lm -o %t
// RUN: %t
// RUN: opt -O3 -S %t.smooth.ll -o %t.smooth.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.smooth.opt.ll -o %t.smooth.opt.o
// RUN: opt -O3 -S %t.self.ll -o %t.self.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.self.opt.ll -o %t.self.opt.o
// RUN: opt -O3 -S %t.while.ll -o %t.while.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.while.opt.ll -o %t.while.opt.o
// RUN: opt -O3 -S %t.branch.ll -o %t.branch.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.branch.opt.ll -o %t.branch.opt.o
// RUN: opt -O3 -S %t.swap.ll -o %t.swap.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.swap.opt.ll -o %t.swap.opt.o
// RUN: opt -O3 -S %t.alias.ll -o %t.alias.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.alias.opt.ll -o %t.alias.opt.o
// RUN: opt -O3 -S %t.nest.ll -o %t.nest.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.nest.opt.ll -o %t.nest.opt.o
// RUN: opt -O3 -S %t.q15.ll -o %t.q15.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.q15.opt.ll -o %t.q15.opt.o
// RUN: opt -O3 -S %t.read.ll -o %t.read.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.read.opt.ll -o %t.read.opt.o
// RUN: opt -O3 -S %t.outer.ll -o %t.outer.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.outer.opt.ll -o %t.outer.opt.o
// RUN: opt -O3 -S %t.mix.ll -o %t.mix.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.mix.opt.ll -o %t.mix.opt.o
// RUN: opt -O3 -S %t.pick.ll -o %t.pick.opt.ll
// RUN: opt -O3 -S %t.named.ll -o %t.named.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.pick.opt.ll -o %t.pick.opt.o
// RUN: llc -relocation-model=pic -filetype=obj %t.named.opt.ll -o %t.named.opt.o
// RUN: cc -std=gnu11 -O2 -ffp-contract=off -include %t.smooth.h -include %t.self.h -include %t.while.h -include %t.branch.h -include %t.swap.h -include %t.alias.h -include %t.nest.h -include %t.q15.h -include %t.read.h -include %t.outer.h -include %t.mix.h -include %t.pick.h -include %t.named.h %S/Inputs/ox_tensor_carry_aot.c %t.smooth.opt.o %t.self.opt.o %t.while.opt.o %t.branch.opt.o %t.swap.opt.o %t.alias.opt.o %t.nest.opt.o %t.q15.opt.o %t.read.opt.o %t.outer.opt.o %t.mix.opt.o %t.pick.opt.o %t.named.opt.o -lm -o %t.opt
// RUN: %t.opt
// RUN: cat %t.*.ll | FileCheck %s
// CHECK-NOT: call {{.*}}@malloc

// Tensors carried by for, while and if against C references at LLVM O0 and O3,
// with aliases of carried state taken before it is rewritten.

// RUN: ondrix-compile %S/Inputs/ox_br_clip.ox --emit=llvmir --llvm-opt-level=0 > %t.clip.ll
// RUN: ondrix-compile %S/Inputs/ox_br_clip.ox --emit=c-header > %t.clip.h
// RUN: llc -relocation-model=pic -filetype=obj %t.clip.ll -o %t.clip.o
// RUN: ondrix-compile %S/Inputs/ox_br_peak.ox --emit=llvmir --llvm-opt-level=0 > %t.peak.ll
// RUN: ondrix-compile %S/Inputs/ox_br_peak.ox --emit=c-header > %t.peak.h
// RUN: llc -relocation-model=pic -filetype=obj %t.peak.ll -o %t.peak.o
// RUN: ondrix-compile %S/Inputs/ox_br_diff.ox --emit=llvmir --llvm-opt-level=0 > %t.diff.ll
// RUN: ondrix-compile %S/Inputs/ox_br_diff.ox --emit=c-header > %t.diff.h
// RUN: llc -relocation-model=pic -filetype=obj %t.diff.ll -o %t.diff.o
// RUN: ondrix-compile %S/Inputs/ox_br_gcd.ox --emit=llvmir --llvm-opt-level=0 > %t.gcd.ll
// RUN: ondrix-compile %S/Inputs/ox_br_gcd.ox --emit=c-header > %t.gcd.h
// RUN: llc -relocation-model=pic -filetype=obj %t.gcd.ll -o %t.gcd.o
// RUN: ondrix-compile %S/Inputs/ox_br_ratio.ox --emit=llvmir --llvm-opt-level=0 > %t.ratio.ll
// RUN: ondrix-compile %S/Inputs/ox_br_ratio.ox --emit=c-header > %t.ratio.h
// RUN: llc -relocation-model=pic -filetype=obj %t.ratio.ll -o %t.ratio.o
// RUN: ondrix-compile %S/Inputs/ox_br_mix.ox --emit=llvmir --llvm-opt-level=0 > %t.mix.ll
// RUN: ondrix-compile %S/Inputs/ox_br_mix.ox --emit=c-header > %t.mix.h
// RUN: llc -relocation-model=pic -filetype=obj %t.mix.ll -o %t.mix.o
// RUN: ondrix-compile %S/Inputs/ox_br_window.ox --emit=llvmir --llvm-opt-level=0 > %t.window.ll
// RUN: ondrix-compile %S/Inputs/ox_br_window.ox --emit=c-header > %t.window.h
// RUN: llc -relocation-model=pic -filetype=obj %t.window.ll -o %t.window.o
// RUN: cc -std=gnu11 -O2 -ffp-contract=off -include %t.clip.h -include %t.peak.h -include %t.diff.h -include %t.gcd.h -include %t.ratio.h -include %t.mix.h -include %t.window.h %S/Inputs/ox_branches_aot.c %t.clip.o %t.peak.o %t.diff.o %t.gcd.o %t.ratio.o %t.mix.o %t.window.o -o %t
// RUN: %t
// RUN: opt -O3 -S %t.clip.ll -o %t.clip.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.clip.opt.ll -o %t.clip.opt.o
// RUN: opt -O3 -S %t.peak.ll -o %t.peak.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.peak.opt.ll -o %t.peak.opt.o
// RUN: opt -O3 -S %t.diff.ll -o %t.diff.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.diff.opt.ll -o %t.diff.opt.o
// RUN: opt -O3 -S %t.gcd.ll -o %t.gcd.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.gcd.opt.ll -o %t.gcd.opt.o
// RUN: opt -O3 -S %t.ratio.ll -o %t.ratio.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ratio.opt.ll -o %t.ratio.opt.o
// RUN: opt -O3 -S %t.mix.ll -o %t.mix.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.mix.opt.ll -o %t.mix.opt.o
// RUN: opt -O3 -S %t.window.ll -o %t.window.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.window.opt.ll -o %t.window.opt.o
// RUN: cc -std=gnu11 -O2 -ffp-contract=off -include %t.clip.h -include %t.peak.h -include %t.diff.h -include %t.gcd.h -include %t.ratio.h -include %t.mix.h -include %t.window.h %S/Inputs/ox_branches_aot.c %t.clip.opt.o %t.peak.opt.o %t.diff.opt.o %t.gcd.opt.o %t.ratio.opt.o %t.mix.opt.o %t.window.opt.o -o %t.opt
// RUN: %t.opt

// Branches, while loops and conditional expressions against C references at LLVM O0 and O3,
// with witnesses that an untaken arm or body never runs and a branch forgets constants.

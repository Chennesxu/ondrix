// RUN: ondrix-compile %S/../Frontend/Inputs/q15_dot.ox --checked-entries --emit=llvmir > %t.dot.ll
// RUN: ondrix-compile %S/../Frontend/Inputs/q15_dot.ox --checked-entries --emit=c-header > %t.dot.h
// RUN: ondrix-compile %S/../Frontend/Inputs/q15_multi_use_binding.ox --checked-entries --emit=llvmir > %t.chain.ll
// RUN: ondrix-compile %S/../Frontend/Inputs/q15_multi_use_binding.ox --checked-entries --emit=c-header > %t.chain.h
// RUN: llc -relocation-model=pic -filetype=obj %t.dot.ll -o %t.dot.o
// RUN: llc -relocation-model=pic -filetype=obj %t.chain.ll -o %t.chain.o
// RUN: cc -include %t.dot.h -include %t.chain.h %S/Inputs/ox_c_entry_checked_aot.c %t.dot.o %t.chain.o -o %t
// RUN: %t

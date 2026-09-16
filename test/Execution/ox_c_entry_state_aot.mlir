// RUN: ondrix-compile %S/../Frontend/Inputs/q15_sos_state_entry.ox --emit=llvmir > %t.ll
// RUN: FileCheck %s --input-file=%t.ll
// RUN: ondrix-compile %S/../Frontend/Inputs/q15_sos_state_entry.ox --emit=c-header > %t.h
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc -include %t.h %S/Inputs/ox_c_entry_state_aot.c %t.o -o %t
// RUN: %t

// A stateful kernel through the plain C entry: the next state reaches the
// caller's output buffer, the const input state is untouched, and the object
// carries no libcall for the copy.
// CHECK-NOT: memcpy
// CHECK: define {{.*}} @ondrix_q15_sos_state_entry(
// CHECK-NOT: memcpy

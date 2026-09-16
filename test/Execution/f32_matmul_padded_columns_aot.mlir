// RUN: ondrix-compile %S/../Frontend/Inputs/f32_matmul_narrow_vec.ox --emit=llvmir --vector-bits=256 --supports-f32-vector-fma > %t.nv.ll
// RUN: FileCheck %s --check-prefix=NARROW --input-file=%t.nv.ll
// RUN: ondrix-compile %S/../Frontend/Inputs/f32_matmul_narrow_vec.ox --emit=c-header --vector-bits=256 --supports-f32-vector-fma > %t.nv.h
// RUN: ondrix-compile %S/../Frontend/Inputs/f32_matmul_narrow_ord.ox --emit=llvmir --vector-bits=0 > %t.no.ll
// RUN: ondrix-compile %S/../Frontend/Inputs/f32_matmul_narrow_ord.ox --emit=c-header --vector-bits=0 > %t.no.h
// RUN: ondrix-compile %S/../Frontend/Inputs/f32_matmul_six_vec.ox --emit=llvmir --vector-bits=256 > %t.sv.ll
// RUN: FileCheck %s --check-prefix=SIX --input-file=%t.sv.ll
// RUN: ondrix-compile %S/../Frontend/Inputs/f32_matmul_six_vec.ox --emit=c-header --vector-bits=256 > %t.sv.h
// RUN: ondrix-compile %S/../Frontend/Inputs/f32_matmul_six_ord.ox --emit=llvmir --vector-bits=0 > %t.so.ll
// RUN: ondrix-compile %S/../Frontend/Inputs/f32_matmul_six_ord.ox --emit=c-header --vector-bits=0 > %t.so.h
// RUN: llc -relocation-model=pic -mattr=+avx2,+fma -filetype=obj %t.nv.ll -o %t.nv.o
// RUN: llc -relocation-model=pic -filetype=obj %t.no.ll -o %t.no.o
// RUN: llc -relocation-model=pic -mattr=+avx2 -filetype=obj %t.sv.ll -o %t.sv.o
// RUN: llc -relocation-model=pic -filetype=obj %t.so.ll -o %t.so.o
// RUN: cc -include %t.nv.h -include %t.no.h -include %t.sv.h -include %t.so.h %S/Inputs/f32_matmul_padded_columns_aot.c %t.nv.o %t.no.o %t.sv.o %t.so.o -lm -o %t
// RUN: %t

// The padded block is order preserving, so under the two exact contracts the
// batched and the ordered objects must agree bit for bit, including where the
// surplus lanes read an infinity from the following row. The batched objects
// must actually carry the padded lanes, or the comparison proves nothing.
// NARROW: <4 x float>
// NARROW-NOT: <2 x float>
// SIX: <8 x float>

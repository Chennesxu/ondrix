// RUN: ondrix-opt %s --ondrix-default-pipeline="checked-entries=true" > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-ondrix-c-header > %t.h
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc -include %t.h %S/Inputs/scalar_c_entry_aot.c %t.o -lm -o %t
// RUN: %t
// RUN: opt -O3 -S %t.ll -o %t.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.opt.ll -o %t.opt.o
// RUN: cc -include %t.h %S/Inputs/scalar_c_entry_aot.c %t.opt.o -lm -o %t.opt
// RUN: %t.opt

// Handwritten contracts isolate C entry ABI behavior from frontend parsing.
func.func @scalar_i32(%a: i32 loc("a"), %b: i32 loc("b")) -> i32 attributes {llvm.emit_c_interface} {
  %r = arith.subi %a, %b : i32
  return %r : i32
}
func.func @scalar_i64(%a: i64 loc("a"), %b: i64 loc("b")) -> i64 attributes {llvm.emit_c_interface} {
  %r = arith.subi %a, %b : i64
  return %r : i64
}
func.func @scalar_f32(%a: f32 loc("a")) -> f32 attributes {llvm.emit_c_interface} {
  return %a : f32
}
func.func @scalar_bool(%a: i1 loc("a"), %b: i1 loc("b")) -> i1 attributes {llvm.emit_c_interface} {
  %r = arith.cmpi ne, %a, %b : i1
  return %r : i1
}
func.func @scalar_mixed(%seed: f32 loc("seed"), %lhs: memref<?xf32> loc("lhs"), %flag: i1 loc("flag"), %rhs: memref<?xf32> loc("rhs"), %extra: i64 loc("extra")) -> f32 attributes {llvm.emit_c_interface} {
  %sum = ondsp.reduce_mac %seed, %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = fma>} : (f32, memref<?xf32>, memref<?xf32>) -> f32
  %extra_f = arith.sitofp %extra : i64 to f32
  %total = arith.addf %sum, %extra_f : f32
  %r = arith.select %flag, %total, %seed : f32
  return %r : f32
}
func.func @scalar_fill(%value: i32 loc("value")) -> memref<2xi32> attributes {llvm.emit_c_interface} {
  %buffer = memref.alloc() : memref<2xi32>
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  memref.store %value, %buffer[%zero] : memref<2xi32>
  memref.store %value, %buffer[%one] : memref<2xi32>
  return %buffer : memref<2xi32>
}
func.func @scalar_lengths(%a: memref<?xi32> loc("a"), %bias: i64 loc("bias"), %b: memref<?xi16> loc("b"), %flag: i1 loc("flag")) -> i64 attributes {llvm.emit_c_interface} {
  %zero = arith.constant 0 : index
  %na = memref.dim %a, %zero : memref<?xi32>
  %nb = memref.dim %b, %zero : memref<?xi16>
  %difference = arith.subi %na, %nb : index
  %signed = arith.index_cast %difference : index to i64
  %sum = arith.addi %signed, %bias : i64
  %result = arith.select %flag, %sum, %bias : i64
  return %result : i64
}

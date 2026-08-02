// RUN: ondrix-opt %s --vectorize-ondsp-fp-fast-memref-reduce | FileCheck %s --check-prefix=VECTORIZED
// RUN: ondrix-opt %s --vectorize-ondsp-fp-fast-memref-reduce --convert-scf-to-cf --convert-vector-to-llvm --finalize-memref-to-llvm --convert-math-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc -ffp-contract=off %S/Inputs/fp_fast_reduce_aot.c %t.o -lm -o %t
// RUN: %t

// Object gate for the declared fast relaxation on a rank-1 f32 reduction. The
// result is never bit-pinned, so the executed evidence is an error envelope
// against an f64 reference, bit-identical repeated calls, and a directed
// cancellation corpus on which the result must differ from an ordered scalar
// fma chain. The pass description carries the authorization argument.

// The structural pin runs before the object is built, so a silently
// unvectorized kernel cannot satisfy the envelope by staying scalar.
// VECTORIZED-LABEL: func.func @f32_dot_fast
// VECTORIZED: math.fma {{.*}}fastmath<reassoc,contract> : vector<8xf32>
// VECTORIZED: vector.reduction <add>

// Dynamic extents on purpose: one object serves every trial length.
func.func @f32_dot_fast(%lhs: memref<?xf32>, %rhs: memref<?xf32>) -> f32
    attributes {llvm.emit_c_interface} {
  %zero = arith.constant 0.0 : f32
  %r = ondsp.reduce_mac %zero, %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = fast>} : (f32, memref<?xf32>, memref<?xf32>) -> f32
  return %r : f32
}

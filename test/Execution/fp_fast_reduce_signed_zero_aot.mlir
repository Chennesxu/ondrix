// RUN: ondrix-opt %s --vectorize-ondsp-fp-fast-memref-reduce=vector-width=8 --convert-scf-to-cf --convert-vector-to-llvm --finalize-memref-to-llvm --convert-math-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.sep.mlir
// RUN: ondrix-translate %t.sep.mlir --mlir-to-llvmir > %t.sep.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.sep.ll -o %t.sep.o
// RUN: ondrix-opt %s --vectorize-ondsp-fp-fast-memref-reduce="vector-width=8 supports-vector-fma=true" --convert-scf-to-cf --convert-vector-to-llvm --finalize-memref-to-llvm --convert-math-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.fma.mlir
// RUN: ondrix-translate %t.fma.mlir --mlir-to-llvmir > %t.fma.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.fma.ll -o %t.fma.o
// RUN: cc -ffp-contract=off -DSUFFIX=sep %S/Inputs/fp_fast_reduce_signed_zero_aot.c %t.sep.o -lm -o %t.sep
// RUN: cc -ffp-contract=off -DSUFFIX=fma %S/Inputs/fp_fast_reduce_signed_zero_aot.c %t.fma.o -lm -o %t.fma
// RUN: %t.sep
// RUN: %t.fma

// An all-negative-zero reduction has one value in its whole legal set, so it
// separates a term-conserving schedule from one that synthesizes a start
// value: every real term is -0.0 and -0.0 + -0.0 = -0.0, while any injected
// +0.0 makes the sum +0.0. Both term selections are covered because the
// injected value would sit in the cross-lane fold, which neither selects.
//
// The dynamic kernel runs the control paths a static extent cannot reach: the
// empty reduction, the short branch below one block, the boundary, the tail,
// and a loop iteration plus tail together. Those branches were previously
// covered only by a structural pin, and a structural pin is what encoded this
// defect in the first place.
//
// What this gate shows is narrower than term conservation: no SYNTHESIZED
// IDENTITY is observable. A redundant term that happens to be zero on this
// corpus would pass. Conservation rests on three tracks together — the
// indexed-term bijection in the structure pins, the integer lattice for
// dropped, duplicated or misindexed non-zero terms, and this one for injected
// identities.

func.func @fast_reduce_8(%init: f32, %lhs: memref<8xf32>, %rhs: memref<8xf32>) -> f32
    attributes {llvm.emit_c_interface} {
  %r = ondsp.reduce_mac %init, %lhs, %rhs {
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (f32, memref<8xf32>, memref<8xf32>) -> f32
  return %r : f32
}

func.func @fast_reduce_9(%init: f32, %lhs: memref<9xf32>, %rhs: memref<9xf32>) -> f32
    attributes {llvm.emit_c_interface} {
  %r = ondsp.reduce_mac %init, %lhs, %rhs {
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (f32, memref<9xf32>, memref<9xf32>) -> f32
  return %r : f32
}

func.func @fast_reduce_16(%init: f32, %lhs: memref<16xf32>, %rhs: memref<16xf32>) -> f32
    attributes {llvm.emit_c_interface} {
  %r = ondsp.reduce_mac %init, %lhs, %rhs {
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (f32, memref<16xf32>, memref<16xf32>) -> f32
  return %r : f32
}

func.func @fast_reduce_dynamic(%init: f32, %lhs: memref<?xf32>, %rhs: memref<?xf32>) -> f32
    attributes {llvm.emit_c_interface} {
  %r = ondsp.reduce_mac %init, %lhs, %rhs {
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (f32, memref<?xf32>, memref<?xf32>) -> f32
  return %r : f32
}

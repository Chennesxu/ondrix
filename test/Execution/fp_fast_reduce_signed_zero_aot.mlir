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

// An all-negative-zero reduction has one value in its legal set, so the sign bit
// separates conservation from a synthesized start (ledger: what this cannot show).

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

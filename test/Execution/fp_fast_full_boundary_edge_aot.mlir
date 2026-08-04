// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --cse --canonicalize --vectorize-ondsp-fp-fast-memref-reduce="vector-width=8" --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-vector-to-llvm --finalize-memref-to-llvm --convert-math-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.sep.mlir
// RUN: ondrix-translate %t.sep.mlir --mlir-to-llvmir > %t.sep.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.sep.ll -o %t.sep.o
// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --cse --canonicalize --vectorize-ondsp-fp-fast-memref-reduce="vector-width=8 supports-vector-fma=true" --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-vector-to-llvm --finalize-memref-to-llvm --convert-math-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.fma.mlir
// RUN: ondrix-translate %t.fma.mlir --mlir-to-llvmir > %t.fma.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.fma.ll -o %t.fma.o
// RUN: cc -ffp-contract=off %S/Inputs/fp_fast_full_boundary_edge_aot.c %t.sep.o -lm -o %t.sep
// RUN: cc -ffp-contract=off %S/Inputs/fp_fast_full_boundary_edge_aot.c %t.fma.o -lm -o %t.fma
// RUN: %t.sep
// RUN: %t.fma

// The executed leg of the mixed route. A skipped tap is only distinguishable
// from a zero-padded one at an infinity, which no existing corpus carries.

func.func @f32_fast_full(%input: tensor<?xf32>, %coeffs: tensor<?xf32>, %init: tensor<?xf32>)
    -> tensor<?xf32> attributes {llvm.emit_c_interface} {
  %r = ondrix.fir_filter %input, %coeffs, %init {
    boundary = #ondrix.fir_boundary<full>,
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %r : tensor<?xf32>
}

// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/moving_average_q15_aot.c %t.o -o %t
// RUN: %t

// Valid-boundary moving average: exact window sums with one nearest-even
// rounding per output. The independent reference also witnesses that the
// equal-tap Q15 FIR reformulation (quantize 1/K first) is a different
// program, pinning why the contract is exact-sum-then-round.

func.func @moving_average8_q15(%input: tensor<40xi16>) -> tensor<33xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.moving_average %input {
    window = 8 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : (tensor<40xi16>) -> tensor<33xi16>
  return %result : tensor<33xi16>
}

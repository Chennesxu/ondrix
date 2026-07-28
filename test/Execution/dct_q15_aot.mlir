// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/dct_q15_aot.c %t.o -o %t
// RUN: %t

// Unnormalized type-II DCT at three extents (including the maximum, 64) against an independent exact
// reference embedding mpmath-derived coefficient tables. The i64 sums are
// exact and the single boundary per output is the nearest-even export to
// the frac = 14 - log2(N) reading, whose saturation is provably
// unreachable (|X[k]| <= 16383).

func.func @dct8_q15(%input: tensor<8xi16>) -> tensor<8xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 11>
  } : (tensor<8xi16>) -> tensor<8xi16>
  return %result : tensor<8xi16>
}

func.func @dct32_q15(%input: tensor<32xi16>) -> tensor<32xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 9>
  } : (tensor<32xi16>) -> tensor<32xi16>
  return %result : tensor<32xi16>
}

func.func @dct64_q15(%input: tensor<64xi16>) -> tensor<64xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 8>
  } : (tensor<64xi16>) -> tensor<64xi16>
  return %result : tensor<64xi16>
}

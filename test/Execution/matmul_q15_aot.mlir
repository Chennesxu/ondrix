// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/matmul_q15_aot.c %t.o -o %t
// RUN: %t

// Q15 matrix product against an independent exact reference. Directed
// corpus: the Q15 identity (32767 diagonal is NOT an exact identity —
// scaling by 32767/32768 rounds most magnitudes down by one output LSB
// only at full scale; the reference decides, not intuition), all-minimum
// against all-minimum (every element saturates: K * 2^30 sums), single
// row/column extremes, and deterministic random trials. Rectangular
// shapes exercise all three independent dimensions.

func.func @matmul8x8x8_q15(%a: tensor<8x8xi16>, %b: tensor<8x8xi16>) -> tensor<8x8xi16>
    attributes {llvm.emit_c_interface} {
  %c = ondrix.matmul %a, %b {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8x8xi16>, tensor<8x8xi16>) -> tensor<8x8xi16>
  return %c : tensor<8x8xi16>
}

func.func @matmul4x16x3_q15(%a: tensor<4x16xi16>, %b: tensor<16x3xi16>) -> tensor<4x3xi16>
    attributes {llvm.emit_c_interface} {
  %c = ondrix.matmul %a, %b {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4x16xi16>, tensor<16x3xi16>) -> tensor<4x3xi16>
  return %c : tensor<4x3xi16>
}

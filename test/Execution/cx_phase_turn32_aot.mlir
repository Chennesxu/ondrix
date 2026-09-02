// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-math-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/cx_phase_turn32_aot.c %t.o -lm -o %t
// RUN: %t

// The Q0.32 turn at both component widths: bit-exact against a restatement of
// the angle-addition program, within one LSB of the correctly rounded turn,
// and exact on the axes and diagonals, which are exact turn arithmetic.

func.func @phase_q15_turn32(%input: tensor<4096xi32>) -> tensor<4096xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.cx_phase %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<unsigned, storage = i32, frac = 32>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4096xi32>) -> tensor<4096xi32>
  return %result : tensor<4096xi32>
}

func.func @phase_q31_turn32(%input: tensor<4096xi64>) -> tensor<4096xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.cx_phase %input {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    output_numeric = #ondsp.fixed<unsigned, storage = i32, frac = 32>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4096xi64>) -> tensor<4096xi32>
  return %result : tensor<4096xi32>
}

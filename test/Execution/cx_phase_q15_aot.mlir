// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-math-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/cx_phase_q15_aot.c %t.o -lm -o %t
// RUN: %t

// The input domain is 2^32 pairs, so this gate is structured rather than
// exhaustive and says so: the axes and diagonals over their whole range, one
// component swept whole against ten pivots of the other, a deterministic
// sweep, and the fifteen angles the contract names directly.

func.func @cx_phase_q15(%input: tensor<4096xi32>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.cx_phase %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<unsigned, storage = i16, frac = 16>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4096xi32>) -> tensor<4096xi16>
  return %result : tensor<4096xi16>
}

// The Q31 arm shares the turn reading, so the named angles are the same
// numbers; only the component width and the packed container change.
func.func @cx_phase_q31(%input: tensor<10xi64>) -> tensor<10xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.cx_phase %input {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    output_numeric = #ondsp.fixed<unsigned, storage = i16, frac = 16>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<10xi64>) -> tensor<10xi16>
  return %result : tensor<10xi16>
}

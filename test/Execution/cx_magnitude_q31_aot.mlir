// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/cx_magnitude_q31_aot.c %t.o -o %t -lm
// RUN: %t
// RUN: ondrix-opt %s --ondrix-default-pipeline="vector-bits=256" > %t.pipeline.mlir
// RUN: ondrix-translate %t.pipeline.mlir --mlir-to-llvmir > %t.pipeline.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.pipeline.ll -o %t.pipeline.o
// RUN: cc %S/Inputs/cx_magnitude_q31_aot.c %t.pipeline.o -o %t.pipeline -lm
// RUN: %t.pipeline

// The corner the boundary exists for is re = im = INT32_MIN, whose exact sum
// of squares is 2^63 -- one past the signed i64 maximum. The two rounding arms
// must not agree everywhere, which a pinned component rounding would fail.

func.func @magnitude_q31_even(%input: tensor<32xi64>) -> tensor<32xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.cx_magnitude %input {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    input_rounding = #ondsp.rounding<nearest_even>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<32xi64>) -> tensor<32xi32>
  return %result : tensor<32xi32>
}

func.func @magnitude_q31_floor(%input: tensor<32xi64>) -> tensor<32xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.cx_magnitude %input {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    input_rounding = #ondsp.rounding<toward_negative>,
    rounding = #ondsp.rounding<toward_negative>
  } : (tensor<32xi64>) -> tensor<32xi32>
  return %result : tensor<32xi32>
}

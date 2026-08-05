// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: FileCheck %s --check-prefix=LOWERED < %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc -DFIR_INTERPOLATE_TIES_POSITIVE_SYMBOL=_mlir_ciface_fir_interpolate_q15_ties_positive %S/Inputs/fir_interpolate_q15_aot.c %t.o -o %t
// RUN: %t

// LOWERED-LABEL: llvm.func @fir_interpolate_q15
// LOWERED-NOT: ondrix.
// LOWERED-NOT: ondsp.
// LOWERED-NOT: tensor.

func.func @fir_interpolate_q15(
    %input: tensor<4xi16>, %coeffs: tensor<3xi16>) -> tensor<9xi16>
    attributes {llvm.emit_c_interface} {
  %init = tensor.empty() : tensor<9xi16>
  %result = ondrix.fir_interpolate %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    factor = 2,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4xi16>, tensor<3xi16>, tensor<9xi16>) -> tensor<9xi16>
  return %result : tensor<9xi16>
}

func.func @fir_interpolate_q15_ties_positive(
    %input: tensor<4xi16>, %coeffs: tensor<3xi16>) -> tensor<9xi16>
    attributes {llvm.emit_c_interface} {
  %init = tensor.empty() : tensor<9xi16>
  %result = ondrix.fir_interpolate %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    factor = 2,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_ties_positive>
  } : (tensor<4xi16>, tensor<3xi16>, tensor<9xi16>) -> tensor<9xi16>
  return %result : tensor<9xi16>
}

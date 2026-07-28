// RUN: ondrix-opt %s --evaluate-ondrix-fir-design > %t.evaluated.mlir
// RUN: FileCheck %s --check-prefix=EVALUATED < %t.evaluated.mlir
// RUN: ondrix-opt %t.evaluated.mlir --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --cse --canonicalize --convert-ondsp-fixed-to-scalar --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/fir_design_highpass_q15_aot.c %t.o -o %t
// RUN: %t

// The highpass spectral-inversion counterpart of the lowpass design gate:
// the windowed-sinc highpass profile is designed and quantized at compile
// time and the object is checked against an independent C reference that
// embeds the mpmath-derived golden taps. The center tap is
// q15(1 - 0.5) = 16384 and every off-center tap is the negated lowpass
// tap, which the generic scalar consumer must reproduce bit-exactly.

// EVALUATED-LABEL: func.func @fir_design_highpass_q15
// EVALUATED-NOT: ondrix.fir_design_windowed_sinc
// EVALUATED: arith.constant
// EVALUATED-SAME: response = "highpass"
// EVALUATED-SAME: dense<[0, 747, 0, -9025, 16384, -9025, 0, 747, 0]> : tensor<9xi16>

func.func @fir_design_highpass_q15(%input: tensor<64xi16>) -> tensor<56xi16>
    attributes {llvm.emit_c_interface} {
  %coefficients = ondrix.fir_design_windowed_sinc {
    response = #ondrix.fir_design_response<highpass>,
    cutoff_num = 1 : i64, cutoff_den = 4 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : tensor<9xi16>
  %init = tensor.empty() : tensor<56xi16>
  %output = ondrix.fir_filter %input, %coefficients, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<valid>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<64xi16>, tensor<9xi16>, tensor<56xi16>) -> tensor<56xi16>
  return %output : tensor<56xi16>
}

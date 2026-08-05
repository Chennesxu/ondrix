// RUN: ondrix-opt %s --fuse-ondrix-gain-into-fir > %t.fused.ondrix.mlir
// RUN: FileCheck %s --check-prefix=REFUSED < %t.fused.ondrix.mlir
// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/legality_gain_fir_fold_q15_aot.c %t.o -o %t
// RUN: %t

// Object-level replay of the fuse-ondrix-gain-into-fir refusal, both programs
// decided on all 65536 inputs by an independent C reference. The two pairs
// are each other's non-vacuity argument, and perturbing any pinned constant
// by one fails the gate:
//
//   tap 16385 : exactly one diverging input, x = -32768, exact terms
//               536887295 (gain path) versus 536903680 (folded), exported
//               16384 versus 16385 — one observable LSB.
//   tap  9025 : the same single-input term divergence (295722175 versus
//               295731200) is hidden by this export policy, which rounds
//               both to 9025 — observationally identical programs.

// One tap makes the attribution total: every input is observed at exactly one
// output, and the certificate is per tap.

// REFUSED-LABEL: func.func @negate_then_fir_sharp_q15
// REFUSED: ondrix.gain
// REFUSED-SAME: gain = -32768
// REFUSED: arith.constant dense<16385>
// REFUSED: ondrix.fir_filter
// REFUSED-NOT: gain_fusion_provenance
func.func @negate_then_fir_sharp_q15(%input: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %negated = ondrix.gain %input {
    gain = -32768 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4096xi16>) -> tensor<4096xi16>
  %coeffs = arith.constant dense<16385> : tensor<1xi16>
  %init = tensor.empty() : tensor<4096xi16>
  %result = ondrix.fir_filter %negated, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<valid>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4096xi16>, tensor<1xi16>, tensor<4096xi16>) -> tensor<4096xi16>
  return %result : tensor<4096xi16>
}

// The naive folded program, written out directly as the rewrite would have
// produced it. It is a perfectly legal program on its own; it is simply not
// the same program as the one above.
func.func @fir_negated_tap_sharp_q15(%input: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %coeffs = arith.constant dense<-16385> : tensor<1xi16>
  %init = tensor.empty() : tensor<4096xi16>
  %result = ondrix.fir_filter %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<valid>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4096xi16>, tensor<1xi16>, tensor<4096xi16>) -> tensor<4096xi16>
  return %result : tensor<4096xi16>
}

// REFUSED-LABEL: func.func @negate_then_fir_hidden_q15
// REFUSED: ondrix.gain
// REFUSED-SAME: gain = -32768
// REFUSED: arith.constant dense<9025>
// REFUSED: ondrix.fir_filter
// REFUSED-NOT: gain_fusion_provenance
func.func @negate_then_fir_hidden_q15(%input: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %negated = ondrix.gain %input {
    gain = -32768 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4096xi16>) -> tensor<4096xi16>
  %coeffs = arith.constant dense<9025> : tensor<1xi16>
  %init = tensor.empty() : tensor<4096xi16>
  %result = ondrix.fir_filter %negated, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<valid>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4096xi16>, tensor<1xi16>, tensor<4096xi16>) -> tensor<4096xi16>
  return %result : tensor<4096xi16>
}

func.func @fir_negated_tap_hidden_q15(%input: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %coeffs = arith.constant dense<-9025> : tensor<1xi16>
  %init = tensor.empty() : tensor<4096xi16>
  %result = ondrix.fir_filter %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<valid>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4096xi16>, tensor<1xi16>, tensor<4096xi16>) -> tensor<4096xi16>
  return %result : tensor<4096xi16>
}

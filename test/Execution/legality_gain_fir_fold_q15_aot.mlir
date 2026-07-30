// RUN: ondrix-opt %s --fuse-ondrix-gain-into-fir > %t.fused.ondrix.mlir
// RUN: FileCheck %s --check-prefix=REFUSED < %t.fused.ondrix.mlir
// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/legality_gain_fir_fold_q15_aot.c %t.o -o %t
// RUN: %t

// Legality counterexample for folding a constant gain into constant FIR
// taps. In exact real arithmetic, negating the samples and negating the taps
// are the same linear map, so rewriting
//
//   fir_filter(gain(x, -32768), h)   ->   fir_filter(x, -h)
//
// is a valid identity and every compiler textbook performs it. Under the
// contract it is illegal: the gain requantizes and SATURATES every sample at
// its own Q1.15 boundary before the taps see it, and at x = -32768 that
// boundary returns 32767 rather than 32768. The opt-in
// `fuse-ondrix-gain-into-fir` pass refuses the rewrite for exactly that
// reason; this gate replays the refusal at object level over the whole
// 65536-value input domain.
//
// Both programs below are compiled through the ordinary pipeline and each is
// decided, on every one of the 65536 inputs, by an independent C reference.
// The harness then pins the divergence census the certificate predicts:
//
//   tap 16385 : the two programs differ on EXACTLY ONE of 65536 inputs,
//               x = -32768, where the exact accumulator terms are
//               32767 * 16385 = 536887295 (gain path) and
//               32768 * 16385 = 536903680 (folded path), exported to
//               16384 and 16385 — one observable LSB.
//   tap  9025 : the same single-input term divergence exists
//               (295722175 versus 295731200) but this export policy rounds
//               both to 9025, so the two programs are observationally
//               IDENTICAL on the whole domain.
//
// The second pair is the reason the pass certifies exact accumulator terms
// rather than exported samples. A term-level certificate holds for every
// accumulator width, update overflow policy, and export policy at once; an
// output-level check would have declared the 9025 fold legal and then broken
// the moment the surrounding policy changed.
//
// The two pairs are each other's non-vacuity argument: the sharp pair proves
// the harness can see a one-LSB divergence, the hidden pair proves it is not
// reporting divergence unconditionally. Perturbing any pinned constant by one
// — the witness input, either exact term, either exported value, or either
// expected divergence count — fails the gate.

// A single tap keeps the attribution total and exact: every input value is
// observed at exactly one output, and the certificate is per tap anyway, so a
// one-tap counterexample refutes the fold for every filter carrying that tap.

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

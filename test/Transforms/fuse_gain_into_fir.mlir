// RUN: ondrix-opt %s --fuse-ondrix-gain-into-fir | FileCheck %s
// RUN: ondrix-opt %s --fuse-ondrix-gain-into-fir="record-refusals=1" | FileCheck %s --check-prefix=REFUSAL

// Fusing a constant gain into constant FIR taps is a textbook
// real-arithmetic identity: scaling the samples and scaling the taps are the
// same linear map. Under the contract they are not, because the gain rounds
// and saturates every sample at its own Q1.15 boundary before the taps see
// it. The rewrite is therefore authorized per tap by an exhaustive
// certificate over all 65536 i16 inputs:
//
//   for every tap i, for every i16 x:
//     applyGainQ15(x, g, rule) * h[i] == x * h'[i]
//
// Both sides are the exact products that enter the accumulator, so a
// certified filter pushes the identical ordered term sequence and the fusion
// is bit-exact for any accumulator width, update overflow policy, export
// policy, and product selection. One failing tap refuses the whole filter.
//
// ---------------------------------------------------------------------
// CENSUS. The candidate set below was swept offline with exactly the
// integer contract this pass implements. Proposals per tap: the quantized
// q15(g*h/2^15) under the declared rule, and the exact g*h/2^15 when that
// division is exact and representable.
//
//   gains  0, +-2^k (k = 0..14), -32768, +-32767, and the gain-cascade
//          witnesses 3125, 32764, 22938, 19661            (38 distinct)
//   taps   0, +-1, +-2, +-3, 4, 8192, +-16384, 2621, 9025,
//          -747, 32767, -32768                            (16 distinct)
//   rules  nearest_even, nearest_ties_positive
//
//   1216 (gain, tap, rule) combinations evaluated
//    106 certified = every combination with tap == 0 (38 gains x 2 rules)
//                    plus every combination with gain == 0 (15 nonzero
//                    taps x 2 rules)
//      0 certified with both the gain and the tap nonzero
//
// A wider sweep confirms the emptiness is structural, not an artifact of
// the candidate list: for h in {1, -1, 2, 9025, -32768} and ALL 65535
// nonzero gains, under both rules, no gain certifies. A nonzero tap forces
// applyGainQ15(x, g, rule) = x * h'/h for every x, an exact linear identity
// that no representable gain satisfies — the unit gain 2^15 is outside the
// declared range [-32768, 32767].
//
// Divergence census for the natural quantized proposal (rule: NE =
// nearest_even, NP = nearest_ties_positive; witness is the first input):
//
//   g       h      h'     rule  diverging inputs  first witness terms
//   -32768      1     -1  both      1 / 65536  x=-32768:  32767 vs  32768
//   -32768     -1      1  both      1 / 65536  x=-32768: -32767 vs -32768
//   -32768   9025  -9025  both      1 / 65536  x=-32768: 295722175 vs 295731200
//    16384      1      0  NE    65533 / 65536  x=-32768: -16384 vs 0
//    16384      1      1  NP    65534 / 65536  x=-32768: -16384 vs -32768
//    16384      2      1  NE    32768 / 65536  x=-32767: -32768 vs -32767
//    16384      2      1  NP    32768 / 65536  x=-32767: -32766 vs -32767
//    32767      1      1  NE    32767 / 65536  x=-32768: -32767 vs -32768
//    32767      1      1  NP    32768 / 65536  x=-32768: -32767 vs -32768
//    22938      1      1  both  65533 / 65536  x=-32768: -22938 vs -32768
//     3125   9025    861  both  65533 / 65536  x=-32768: -28203125 vs -28213248
//
// Plain negation is the sharpest entry: g = -32768 with h' = -h agrees on
// 65535 of 65536 inputs and is still illegal, because the gain saturates
// -32768 to 32767 instead of 32768.
//
// Unlike the gain-cascade merge, whose certified constant pairs genuinely
// differ between the two tie rules, legality here does NOT depend on the
// rule: the certified set is characterized by gain == 0 or tap == 0, and
// both conditions are rule-independent. The sweep found no (g, h) pair whose
// verdict differs between nearest_even and nearest_ties_positive. The pair
// (16384, 2) below is stated under both rules to pin that: same refusal,
// different divergent term.
// ---------------------------------------------------------------------

// A muted signal path certifies for arbitrary taps: every unfused term is
// applyGainQ15(x, 0) * h[i] = 0 and every fused term is x * 0 = 0. The
// requantization boundary disappears and the tap table becomes all zeros.
// CHECK-LABEL: func.func @certified_zero_gain
// CHECK-NOT: ondrix.gain
// CHECK: %[[TAPS:.*]] = arith.constant dense<0> : tensor<3xi16>
// CHECK: ondrix.fir_filter %arg0, %[[TAPS]], %arg1
// CHECK-SAME: certified_taps = 3
// CHECK-SAME: exhaustive_inputs = 65536
// CHECK-SAME: gain = 0
// CHECK-SAME: rounding = "nearest_even"
// CHECK-NOT: ondrix.gain
// REFUSAL-LABEL: func.func @certified_zero_gain
// REFUSAL-NOT: gain_fusion_refusal
func.func @certified_zero_gain(%input: tensor<8xi16>, %init: tensor<6xi16>) -> tensor<6xi16> {
  %muted = ondrix.gain %input {
    gain = 0 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>) -> tensor<8xi16>
  %coeffs = arith.constant dense<[9025, -747, 16384]> : tensor<3xi16>
  %result = ondrix.fir_filter %muted, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<valid>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>, tensor<3xi16>, tensor<6xi16>) -> tensor<6xi16>
  return %result : tensor<6xi16>
}

// The other certified family: an all-zero tap table annihilates every term
// before the gain can matter, so an arbitrary gain drops out. This is the
// only way a NONZERO gain ever leaves the filter.
// CHECK-LABEL: func.func @certified_zero_taps
// CHECK-NOT: ondrix.gain
// CHECK: ondrix.fir_filter %arg0
// CHECK-SAME: gain = 16384
// CHECK-SAME: rounding = "nearest_even"
// CHECK-NOT: ondrix.gain
func.func @certified_zero_taps(%input: tensor<8xi16>, %init: tensor<6xi16>) -> tensor<6xi16> {
  %scaled = ondrix.gain %input {
    gain = 16384 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>) -> tensor<8xi16>
  %coeffs = arith.constant dense<0> : tensor<3xi16>
  %result = ondrix.fir_filter %scaled, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<valid>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>, tensor<3xi16>, tensor<6xi16>) -> tensor<6xi16>
  return %result : tensor<6xi16>
}

// Negate-then-filter versus filter-with-negated-taps. The two programs agree
// on 65535 of the 65536 inputs; the single separating input is the saturation
// point x = -32768, where the gain yields 32767 and tap 9025 gives the exact
// product 295722175 instead of the fused 295731200. One input is enough.
// CHECK-LABEL: func.func @refused_negation
// CHECK: ondrix.gain
// CHECK-SAME: gain = -32768
// CHECK: arith.constant dense<[9025, -747, 16384]>
// CHECK: ondrix.fir_filter
// CHECK-NOT: gain_fusion_provenance
// REFUSAL-LABEL: func.func @refused_negation
// REFUSAL: ondrix.fir_filter
// REFUSAL-SAME: fused_term = 295731200
// REFUSAL-SAME: gain = -32768
// REFUSAL-SAME: gain_term = 295722175
// REFUSAL-SAME: proposed_tap = -9025
// REFUSAL-SAME: tap = 9025
// REFUSAL-SAME: tap_index = 0
// REFUSAL-SAME: witness_input = -32768
func.func @refused_negation(%input: tensor<8xi16>, %init: tensor<6xi16>) -> tensor<6xi16> {
  %negated = ondrix.gain %input {
    gain = -32768 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>) -> tensor<8xi16>
  %coeffs = arith.constant dense<[9025, -747, 16384]> : tensor<3xi16>
  %result = ondrix.fir_filter %negated, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<valid>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>, tensor<3xi16>, tensor<6xi16>) -> tensor<6xi16>
  return %result : tensor<6xi16>
}

// A generic design gain against a designed tap: the quantized fused tap
// q15(3125 * 9025 / 2^15) = 861 diverges on 65533 of the 65536 inputs, first
// at x = -32768 where the exact products are -28203125 and -28213248.
// CHECK-LABEL: func.func @refused_generic_pair
// CHECK: ondrix.gain
// CHECK-SAME: gain = 3125
// CHECK: arith.constant dense<9025>
// CHECK: ondrix.fir_filter
// CHECK-NOT: gain_fusion_provenance
// REFUSAL-LABEL: func.func @refused_generic_pair
// REFUSAL: ondrix.fir_filter
// REFUSAL-SAME: fused_term = -28213248
// REFUSAL-SAME: gain_term = -28203125
// REFUSAL-SAME: proposed_tap = 861
// REFUSAL-SAME: witness_input = -32768
func.func @refused_generic_pair(%input: tensor<8xi16>, %init: tensor<6xi16>) -> tensor<6xi16> {
  %scaled = ondrix.gain %input {
    gain = 3125 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>) -> tensor<8xi16>
  %coeffs = arith.constant dense<9025> : tensor<3xi16>
  %result = ondrix.fir_filter %scaled, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<valid>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>, tensor<3xi16>, tensor<6xi16>) -> tensor<6xi16>
  return %result : tensor<6xi16>
}

// The certificate is per tap and the filter is all-or-nothing: two certified
// zero taps do not buy the third one. The rewrite would have to delete the
// gain for every tap at once, so a single uncertified tap keeps the whole
// filter unfused.
// CHECK-LABEL: func.func @refused_single_nonzero_tap
// CHECK: ondrix.gain
// CHECK-SAME: gain = 16384
// CHECK: arith.constant dense<[0, 1, 0]>
// CHECK: ondrix.fir_filter
// CHECK-NOT: gain_fusion_provenance
// REFUSAL-LABEL: func.func @refused_single_nonzero_tap
// REFUSAL: ondrix.fir_filter
// REFUSAL-SAME: fused_term = 0
// REFUSAL-SAME: gain_term = -16384
// REFUSAL-SAME: proposed_tap = 0
// REFUSAL-SAME: tap = 1
// REFUSAL-SAME: tap_index = 1
// REFUSAL-SAME: witness_input = -32768
func.func @refused_single_nonzero_tap(%input: tensor<8xi16>, %init: tensor<6xi16>) -> tensor<6xi16> {
  %scaled = ondrix.gain %input {
    gain = 16384 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>) -> tensor<8xi16>
  %coeffs = arith.constant dense<[0, 1, 0]> : tensor<3xi16>
  %result = ondrix.fir_filter %scaled, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<valid>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>, tensor<3xi16>, tensor<6xi16>) -> tensor<6xi16>
  return %result : tensor<6xi16>
}

// A gain observed by another consumer is not the filter's private boundary,
// so it is never deleted — even for a gain whose certificate passes.
// CHECK-LABEL: func.func @refused_multi_use_gain
// CHECK: ondrix.gain
// CHECK-SAME: gain = 0
// CHECK: arith.constant dense<[9025, -747, 16384]>
// CHECK: ondrix.fir_filter
// CHECK-NOT: gain_fusion_provenance
// REFUSAL-LABEL: func.func @refused_multi_use_gain
// REFUSAL-NOT: gain_fusion_refusal
func.func @refused_multi_use_gain(%input: tensor<8xi16>, %init: tensor<6xi16>)
    -> (tensor<6xi16>, tensor<8xi16>) {
  %muted = ondrix.gain %input {
    gain = 0 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>) -> tensor<8xi16>
  %coeffs = arith.constant dense<[9025, -747, 16384]> : tensor<3xi16>
  %result = ondrix.fir_filter %muted, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<valid>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>, tensor<3xi16>, tensor<6xi16>) -> tensor<6xi16>
  return %result, %muted : tensor<6xi16>, tensor<8xi16>
}

// Runtime taps carry no certificate. The analysis stops at the missing
// constant and the program keeps its gain, even though a zero gain would
// certify against any constant table.
// CHECK-LABEL: func.func @refused_dynamic_taps
// CHECK: ondrix.gain
// CHECK-SAME: gain = 0
// CHECK: ondrix.fir_filter
// CHECK-NOT: gain_fusion_provenance
// REFUSAL-LABEL: func.func @refused_dynamic_taps
// REFUSAL-NOT: gain_fusion_refusal
func.func @refused_dynamic_taps(
    %input: tensor<8xi16>, %coeffs: tensor<3xi16>, %init: tensor<6xi16>) -> tensor<6xi16> {
  %muted = ondrix.gain %input {
    gain = 0 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>) -> tensor<8xi16>
  %result = ondrix.fir_filter %muted, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<valid>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>, tensor<3xi16>, tensor<6xi16>) -> tensor<6xi16>
  return %result : tensor<6xi16>
}

// ---------------------------------------------------------------------
// The same (16384, 2) pair under both admitted tie rules. The exact fused
// tap 1 exists here — 16384 * 2 / 2^15 is exact — and both rules still
// refuse, on exactly 32768 of 65536 inputs each. The rules disagree about
// the gain's own value at the witness (-32768 versus -32766), yet the
// verdict is identical: the census found no pair whose LEGALITY depends on
// the rule, which is precisely the opposite of the gain-cascade merge.
//
// Only the GAIN's declared rule parameterizes the certificate; the filter's
// own export rounding is a separate boundary the certificate never reaches,
// and the fixed FIR filter contract does not admit ties-toward-positive
// there anyway.
// ---------------------------------------------------------------------

// CHECK-LABEL: func.func @refused_pair_nearest_even
// CHECK: ondrix.gain
// CHECK-SAME: rounding = #ondsp.rounding<nearest_even>
// CHECK: ondrix.fir_filter
// CHECK-NOT: gain_fusion_provenance
// REFUSAL-LABEL: func.func @refused_pair_nearest_even
// REFUSAL: ondrix.fir_filter
// REFUSAL-SAME: fused_term = -32767
// REFUSAL-SAME: gain_term = -32768
// REFUSAL-SAME: proposed_tap = 1
// REFUSAL-SAME: rounding = "nearest_even"
// REFUSAL-SAME: witness_input = -32767
func.func @refused_pair_nearest_even(%input: tensor<8xi16>, %init: tensor<6xi16>) -> tensor<6xi16> {
  %scaled = ondrix.gain %input {
    gain = 16384 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>) -> tensor<8xi16>
  %coeffs = arith.constant dense<2> : tensor<3xi16>
  %result = ondrix.fir_filter %scaled, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<valid>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>, tensor<3xi16>, tensor<6xi16>) -> tensor<6xi16>
  return %result : tensor<6xi16>
}

// CHECK-LABEL: func.func @refused_pair_ties_positive
// CHECK: ondrix.gain
// CHECK-SAME: rounding = #ondsp.rounding<nearest_ties_positive>
// CHECK: ondrix.fir_filter
// CHECK-NOT: gain_fusion_provenance
// REFUSAL-LABEL: func.func @refused_pair_ties_positive
// REFUSAL: ondrix.fir_filter
// REFUSAL-SAME: fused_term = -32767
// REFUSAL-SAME: gain_term = -32766
// REFUSAL-SAME: proposed_tap = 1
// REFUSAL-SAME: rounding = "nearest_ties_positive"
// REFUSAL-SAME: witness_input = -32767
func.func @refused_pair_ties_positive(%input: tensor<8xi16>, %init: tensor<6xi16>) -> tensor<6xi16> {
  %scaled = ondrix.gain %input {
    gain = 16384 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_ties_positive>
  } : (tensor<8xi16>) -> tensor<8xi16>
  %coeffs = arith.constant dense<2> : tensor<3xi16>
  %result = ondrix.fir_filter %scaled, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<valid>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>, tensor<3xi16>, tensor<6xi16>) -> tensor<6xi16>
  return %result : tensor<6xi16>
}

// The certified zero gain also fires against a full-boundary filter and an
// exhaustively certified rewrite carries the same provenance regardless of
// the surrounding policies, which the certificate proves it cannot touch.
// CHECK-LABEL: func.func @certified_zero_gain_full_boundary
// CHECK-NOT: ondrix.gain
// CHECK: ondrix.fir_filter
// CHECK-SAME: boundary = #ondrix.fir_boundary<full>
// CHECK-SAME: certified_taps = 3
// CHECK-SAME: gain = 0
// CHECK-SAME: product = #ondsp.product<high_raw>
func.func @certified_zero_gain_full_boundary(
    %input: tensor<8xi16>, %init: tensor<10xi16>) -> tensor<10xi16> {
  %muted = ondrix.gain %input {
    gain = 0 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>) -> tensor<8xi16>
  %coeffs = arith.constant dense<[9025, -747, 16384]> : tensor<3xi16>
  %result = ondrix.fir_filter %muted, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i32, frac = 14, signed, update_overflow = wrap>,
    boundary = #ondrix.fir_boundary<full>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 14>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<wrap>,
    product = #ondsp.product<high_raw>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>, tensor<3xi16>, tensor<10xi16>) -> tensor<10xi16>
  return %result : tensor<10xi16>
}

// RUN: ondrix-opt %s --fuse-ondrix-elementwise-chains | FileCheck %s

// The verdict table the certificate produces. Every pair below is a
// real-arithmetic identity; which ones survive the contract is decided by a
// declared value, not by the operation, and the certificate decides each
// candidate rather than applying a rule.

#q15 = #ondsp.fixed<signed, storage = i16, frac = 15>

// A wrapping negation is an involution on the whole domain.
// CHECK-LABEL: func.func @negate_wrap_is_an_involution
// CHECK-NEXT: return %arg0
func.func @negate_wrap_is_an_involution(%a: tensor<8xi16>) -> tensor<8xi16> {
  %0 = ondrix.negate %a {numeric = #q15, overflow = #ondsp.overflow<wrap>}
    : (tensor<8xi16>) -> tensor<8xi16>
  %1 = ondrix.negate %0 {numeric = #q15, overflow = #ondsp.overflow<wrap>}
    : (tensor<8xi16>) -> tensor<8xi16>
  return %1 : tensor<8xi16>
}

// A saturating one is not: -32768 negates to 32767 and back to -32767.
// CHECK-LABEL: func.func @negate_saturate_is_not
// CHECK-COUNT-2: ondrix.negate
func.func @negate_saturate_is_not(%a: tensor<8xi16>) -> tensor<8xi16> {
  %0 = ondrix.negate %a {numeric = #q15, overflow = #ondsp.overflow<saturate>}
    : (tensor<8xi16>) -> tensor<8xi16>
  %1 = ondrix.negate %0 {numeric = #q15, overflow = #ondsp.overflow<saturate>}
    : (tensor<8xi16>) -> tensor<8xi16>
  return %1 : tensor<8xi16>
}

// Floor of a floor is a floor, so two directed right shifts are one.
// CHECK-LABEL: func.func @shift_cascade_under_a_directed_rule
// CHECK: ondrix.shift
// CHECK-SAME: amount = -5
// CHECK-SAME: parameter = -5
// CHECK-NOT: ondrix.shift
func.func @shift_cascade_under_a_directed_rule(%a: tensor<8xi16>) -> tensor<8xi16> {
  %0 = ondrix.shift %a {amount = -2 : i64, numeric = #q15,
    rounding = #ondsp.rounding<toward_negative>, overflow = #ondsp.overflow<saturate>}
    : (tensor<8xi16>) -> tensor<8xi16>
  %1 = ondrix.shift %0 {amount = -3 : i64, numeric = #q15,
    rounding = #ondsp.rounding<toward_negative>, overflow = #ondsp.overflow<saturate>}
    : (tensor<8xi16>) -> tensor<8xi16>
  return %1 : tensor<8xi16>
}

// Under a nearest rule the same cascade double rounds and is left alone.
// CHECK-LABEL: func.func @shift_cascade_under_a_nearest_rule
// CHECK-COUNT-2: ondrix.shift
func.func @shift_cascade_under_a_nearest_rule(%a: tensor<8xi16>) -> tensor<8xi16> {
  %0 = ondrix.shift %a {amount = -2 : i64, numeric = #q15,
    rounding = #ondsp.rounding<nearest_even>, overflow = #ondsp.overflow<saturate>}
    : (tensor<8xi16>) -> tensor<8xi16>
  %1 = ondrix.shift %0 {amount = -3 : i64, numeric = #q15,
    rounding = #ondsp.rounding<nearest_even>, overflow = #ondsp.overflow<saturate>}
    : (tensor<8xi16>) -> tensor<8xi16>
  return %1 : tensor<8xi16>
}

// The exception the certificate found and a per-rule verdict would miss:
// nearest_even at amounts -1,-1 composes exactly (0 diverging inputs), while
// nearest_ties_positive diverges on 16384 inputs at the same amounts.
// CHECK-LABEL: func.func @nearest_even_unit_pair_is_the_exception
// CHECK: ondrix.shift
// CHECK-SAME: amount = -2
// CHECK-NOT: ondrix.shift
func.func @nearest_even_unit_pair_is_the_exception(%a: tensor<8xi16>) -> tensor<8xi16> {
  %0 = ondrix.shift %a {amount = -1 : i64, numeric = #q15,
    rounding = #ondsp.rounding<nearest_even>, overflow = #ondsp.overflow<saturate>}
    : (tensor<8xi16>) -> tensor<8xi16>
  %1 = ondrix.shift %0 {amount = -1 : i64, numeric = #q15,
    rounding = #ondsp.rounding<nearest_even>, overflow = #ondsp.overflow<saturate>}
    : (tensor<8xi16>) -> tensor<8xi16>
  return %1 : tensor<8xi16>
}

// CHECK-LABEL: func.func @ties_positive_unit_pair_is_not
// CHECK-COUNT-2: ondrix.shift
func.func @ties_positive_unit_pair_is_not(%a: tensor<8xi16>) -> tensor<8xi16> {
  %0 = ondrix.shift %a {amount = -1 : i64, numeric = #q15,
    rounding = #ondsp.rounding<nearest_ties_positive>, overflow = #ondsp.overflow<saturate>}
    : (tensor<8xi16>) -> tensor<8xi16>
  %1 = ondrix.shift %0 {amount = -1 : i64, numeric = #q15,
    rounding = #ondsp.rounding<nearest_ties_positive>, overflow = #ondsp.overflow<saturate>}
    : (tensor<8xi16>) -> tensor<8xi16>
  return %1 : tensor<8xi16>
}

// Two left shifts collapse under either overflow mode.
// CHECK-LABEL: func.func @left_shift_cascade
// CHECK: ondrix.shift
// CHECK-SAME: amount = 5
// CHECK-NOT: ondrix.shift
func.func @left_shift_cascade(%a: tensor<8xi16>) -> tensor<8xi16> {
  %0 = ondrix.shift %a {amount = 2 : i64, numeric = #q15,
    rounding = #ondsp.rounding<nearest_even>, overflow = #ondsp.overflow<saturate>}
    : (tensor<8xi16>) -> tensor<8xi16>
  %1 = ondrix.shift %0 {amount = 3 : i64, numeric = #q15,
    rounding = #ondsp.rounding<nearest_even>, overflow = #ondsp.overflow<saturate>}
    : (tensor<8xi16>) -> tensor<8xi16>
  return %1 : tensor<8xi16>
}

// Wrapping bias is a group action; saturating bias is not, and +1000 then
// -1000 loses exactly 1000 at the rail.
// CHECK-LABEL: func.func @offset_cascade_under_wrap
// CHECK-NEXT: return %arg0
func.func @offset_cascade_under_wrap(%a: tensor<8xi16>) -> tensor<8xi16> {
  %0 = ondrix.offset %a {bias = 1000 : i64, numeric = #q15, overflow = #ondsp.overflow<wrap>}
    : (tensor<8xi16>) -> tensor<8xi16>
  %1 = ondrix.offset %0 {bias = -1000 : i64, numeric = #q15, overflow = #ondsp.overflow<wrap>}
    : (tensor<8xi16>) -> tensor<8xi16>
  return %1 : tensor<8xi16>
}

// CHECK-LABEL: func.func @offset_cascade_under_saturate
// CHECK-COUNT-2: ondrix.offset
func.func @offset_cascade_under_saturate(%a: tensor<8xi16>) -> tensor<8xi16> {
  %0 = ondrix.offset %a {bias = 1000 : i64, numeric = #q15, overflow = #ondsp.overflow<saturate>}
    : (tensor<8xi16>) -> tensor<8xi16>
  %1 = ondrix.offset %0 {bias = -1000 : i64, numeric = #q15, overflow = #ondsp.overflow<saturate>}
    : (tensor<8xi16>) -> tensor<8xi16>
  return %1 : tensor<8xi16>
}

// CHECK-LABEL: func.func @absolute_value_absorbs_negation
// CHECK: ondrix.abs
// CHECK-NOT: ondrix.negate
func.func @absolute_value_absorbs_negation(%a: tensor<8xi16>) -> tensor<8xi16> {
  %0 = ondrix.negate %a {numeric = #q15, overflow = #ondsp.overflow<saturate>}
    : (tensor<8xi16>) -> tensor<8xi16>
  %1 = ondrix.abs %0 {numeric = #q15, overflow = #ondsp.overflow<saturate>}
    : (tensor<8xi16>) -> tensor<8xi16>
  return %1 : tensor<8xi16>
}

// Three shifts reach a fixpoint through two independently certified steps.
// CHECK-LABEL: func.func @three_shifts_reach_a_fixpoint
// CHECK: ondrix.shift
// CHECK-SAME: amount = -6
// CHECK-NOT: ondrix.shift
func.func @three_shifts_reach_a_fixpoint(%a: tensor<8xi16>) -> tensor<8xi16> {
  %0 = ondrix.shift %a {amount = -2 : i64, numeric = #q15,
    rounding = #ondsp.rounding<toward_negative>, overflow = #ondsp.overflow<saturate>}
    : (tensor<8xi16>) -> tensor<8xi16>
  %1 = ondrix.shift %0 {amount = -3 : i64, numeric = #q15,
    rounding = #ondsp.rounding<toward_negative>, overflow = #ondsp.overflow<saturate>}
    : (tensor<8xi16>) -> tensor<8xi16>
  %2 = ondrix.shift %1 {amount = -1 : i64, numeric = #q15,
    rounding = #ondsp.rounding<toward_negative>, overflow = #ondsp.overflow<saturate>}
    : (tensor<8xi16>) -> tensor<8xi16>
  return %2 : tensor<8xi16>
}

// An intermediate value with a second reader is not a chain: collapsing it
// would delete a boundary the other reader observes.
// CHECK-LABEL: func.func @a_second_reader_blocks_the_chain
// CHECK-COUNT-2: ondrix.negate
func.func @a_second_reader_blocks_the_chain(%a: tensor<8xi16>)
    -> (tensor<8xi16>, tensor<8xi16>) {
  %0 = ondrix.negate %a {numeric = #q15, overflow = #ondsp.overflow<wrap>}
    : (tensor<8xi16>) -> tensor<8xi16>
  %1 = ondrix.negate %0 {numeric = #q15, overflow = #ondsp.overflow<wrap>}
    : (tensor<8xi16>) -> tensor<8xi16>
  return %1, %0 : tensor<8xi16>, tensor<8xi16>
}

// A binary member is refused structurally, not by a failed certificate: the
// enumeration that decides every case above cannot be run over 2^32 unknown
// operand pairs.
// CHECK-LABEL: func.func @a_product_cascade_is_never_a_candidate
// CHECK-COUNT-2: ondrix.mult
func.func @a_product_cascade_is_never_a_candidate(%a: tensor<8xi16>, %b: tensor<8xi16>)
    -> tensor<8xi16> {
  %0 = ondrix.mult %a, %b {numeric = #q15, rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>} : (tensor<8xi16>, tensor<8xi16>) -> tensor<8xi16>
  %1 = ondrix.mult %0, %b {numeric = #q15, rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>} : (tensor<8xi16>, tensor<8xi16>) -> tensor<8xi16>
  return %1 : tensor<8xi16>
}

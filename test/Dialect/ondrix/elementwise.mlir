// RUN: ondrix-opt %s | ondrix-opt | FileCheck %s

// CHECK-LABEL: func.func @elementwise_family
// CHECK: ondrix.add
// CHECK-SAME: overflow = #ondsp.overflow<saturate>
// CHECK: ondrix.sub
// CHECK-SAME: overflow = #ondsp.overflow<wrap>
// CHECK: ondrix.mult
// CHECK-SAME: rounding = #ondsp.rounding<nearest_ties_positive>
// CHECK: ondrix.abs
// CHECK: ondrix.negate
// CHECK: ondrix.offset
// CHECK-SAME: bias = -32768
// CHECK: ondrix.shift
// CHECK-SAME: amount = -15
func.func @elementwise_family(%a: tensor<8xi16>, %b: tensor<8xi16>) -> tensor<8xi16> {
  %0 = ondrix.add %a, %b {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>
  } : (tensor<8xi16>, tensor<8xi16>) -> tensor<8xi16>
  %1 = ondrix.sub %0, %b {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<wrap>
  } : (tensor<8xi16>, tensor<8xi16>) -> tensor<8xi16>
  %2 = ondrix.mult %1, %b {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_ties_positive>,
    overflow = #ondsp.overflow<saturate>
  } : (tensor<8xi16>, tensor<8xi16>) -> tensor<8xi16>
  %3 = ondrix.abs %2 {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>
  } : (tensor<8xi16>) -> tensor<8xi16>
  %4 = ondrix.negate %3 {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<wrap>
  } : (tensor<8xi16>) -> tensor<8xi16>
  %5 = ondrix.offset %4 {
    bias = -32768 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>
  } : (tensor<8xi16>) -> tensor<8xi16>
  %6 = ondrix.shift %5 {
    amount = -15 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<saturate>
  } : (tensor<8xi16>) -> tensor<8xi16>
  return %6 : tensor<8xi16>
}

// The rounding mode is declared even where the amount makes it vacuous, so
// changing the amount never silently changes which tie rule applies.
// CHECK-LABEL: func.func @shift_left_still_declares_a_tie_rule
// CHECK: ondrix.shift
// CHECK-SAME: amount = 15
// CHECK-SAME: rounding = #ondsp.rounding<nearest_even>
func.func @shift_left_still_declares_a_tie_rule(%a: tensor<4096xi16>) -> tensor<4096xi16> {
  %0 = ondrix.shift %a {
    amount = 15 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (tensor<4096xi16>) -> tensor<4096xi16>
  return %0 : tensor<4096xi16>
}

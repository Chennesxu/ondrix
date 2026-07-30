// RUN: ondrix-opt %s | ondrix-opt | FileCheck %s

// The lane count is a defaulted type parameter, so single-lane assembly must
// remain byte-identical to what it was before lanes existed: the parameter is
// printed only when it is greater than one. The pins below close the printed
// type with `>` immediately after `update_overflow`, which is exactly the
// absence of a printed lane count.

// CHECK-LABEL: func.func @single_lane_prints_no_lanes
func.func @single_lane_prints_no_lanes(%value: i16, %coefficient: i16)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  // CHECK: ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate>
  %zero = ondsp.acc_zero
      : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  // CHECK: ondsp.mac
  // CHECK-SAME: (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16)
  %result = ondsp.mac %zero, %value, %coefficient {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16)
      -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// Exactly three operations accept more than one lane. The value operand of
// `mac` carries one element per lane while the coefficient stays scalar,
// because the per-lane broadcast is declared semantics rather than a lowering
// detail; `acc_export` produces one destination element per lane.

// CHECK-LABEL: func.func @multi_lane_round_trip
func.func @multi_lane_round_trip(%value: vector<8xi16>, %coefficient: i16) -> vector<8xi16> {
  // CHECK: %[[ZERO:.*]] = ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>
  %zero = ondsp.acc_zero
      : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>
  // CHECK: %[[ACC:.*]] = ondsp.mac %[[ZERO]],
  // CHECK-SAME: (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>, vector<8xi16>, i16)
  // CHECK-SAME: -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>
  %accumulator = ondsp.mac %zero, %value, %coefficient {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>,
       vector<8xi16>, i16)
      -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>
  // CHECK: ondsp.acc_export %[[ACC]]
  // CHECK-SAME: -> vector<8xi16>
  %result = ondsp.acc_export %accumulator {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>)
      -> vector<8xi16>
  return %result : vector<8xi16>
}

// A declared lane count of one is the same type as no declared lane count, so
// it must print without the parameter and remain interchangeable with it.

// CHECK-LABEL: func.func @explicit_one_lane_is_the_default
// CHECK-SAME: -> !ondsp.acc<storage = i34, frac = 30, signed, update_overflow = wrap>
func.func @explicit_one_lane_is_the_default()
    -> !ondsp.acc<storage = i34, frac = 30, signed, update_overflow = wrap> {
  // CHECK: ondsp.acc_zero : <storage = i34, frac = 30, signed, update_overflow = wrap>
  %zero = ondsp.acc_zero
      : !ondsp.acc<storage = i34, frac = 30, signed, update_overflow = wrap, lanes = 1>
  return %zero : !ondsp.acc<storage = i34, frac = 30, signed, update_overflow = wrap>
}

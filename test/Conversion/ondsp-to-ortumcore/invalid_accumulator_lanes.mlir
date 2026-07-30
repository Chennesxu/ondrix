// RUN: not ondrix-opt %s --convert-ondsp-to-ortumcore 2>&1 | FileCheck %s

// The target accumulator type carries no lane count, so a multi-lane source
// accumulator would have its lane count silently dropped on the way to the
// parameterless target type. The lane count is therefore part of the
// accumulator capability gate, not a detail the conversion may ignore.

// CHECK: unsupported accumulator type '!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>'
// CHECK-SAME: ortumcore lowering currently requires !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
func.func @multi_lane_accumulator(%value: vector<8xi16>, %coefficient: i16)
    -> vector<8xi16> {
  %zero = ondsp.acc_zero
      : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>
  %accumulator = ondsp.mac %zero, %value, %coefficient {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>,
       vector<8xi16>, i16)
      -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>
  %result = ondsp.acc_export %accumulator {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>)
      -> vector<8xi16>
  return %result : vector<8xi16>
}

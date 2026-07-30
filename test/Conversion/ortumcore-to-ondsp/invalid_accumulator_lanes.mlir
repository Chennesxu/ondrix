// RUN: not ondrix-opt %s --convert-ortumcore-to-ondsp-emulation 2>&1 | FileCheck %s

// Ondsp operations stay legal in this pass, so a multi-lane accumulator already
// present in the module would flow through untouched next to the single-lane
// accumulators the emulation produces. The emulation profile has no multi-lane
// form, so the module is refused rather than left with two lane meanings mixed.

// CHECK: multi-lane accumulator type '!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>'
// CHECK-SAME: has no OrtumCore emulation profile
func.func @emulation_beside_batched_lanes(
    %lhs: i16, %rhs: i16, %value: vector<8xi16>, %coefficient: i16) -> vector<8xi16> {
  %target = ortumcore.acc_init : !ortumcore.acc
  %updated = ortumcore.mac_add %target, %lhs, %rhs
      : (!ortumcore.acc, i16, i16) -> !ortumcore.acc
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

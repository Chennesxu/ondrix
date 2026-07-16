// RUN: not ondrix-opt %s --convert-ondsp-to-ortumcore 2>&1 | FileCheck %s

func.func @q31_high_mac_is_unproven(
    %acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    %a: i32, %b: i32)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  // CHECK: q31 raw-high target equivalence is not specified
  %0 = ondsp.mac %acc, %a, %b {numeric = #ondsp.fixed<signed, storage = i32, frac = 31>, product = #ondsp.product<high_raw>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i32, i32) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %0 : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

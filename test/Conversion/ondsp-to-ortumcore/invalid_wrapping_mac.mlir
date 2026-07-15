// RUN: not ondrix-opt %s --convert-ondsp-to-ortumcore 2>&1 | FileCheck %s

func.func @wrapping_mac_has_no_target_equivalent(
    %acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>,
    %a: i16, %b: i16)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap> {
  // CHECK: unsupported accumulator type '!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>'
  %0 = ondsp.mac %acc, %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  return %0 : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
}

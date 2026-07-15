// RUN: not ondrix-opt %s --convert-ondsp-to-ortumcore 2>&1 | FileCheck %s

func.func @q15_high_product_is_not_a_full_product_target(
    %acc: !ondsp.acc<storage = i40, frac = 14, signed>, %a: i16, %b: i16)
    -> !ondsp.acc<storage = i40, frac = 14, signed> {
  // CHECK: unsupported accumulator type '!ondsp.acc<storage = i40, frac = 14, signed>'
  %0 = ondsp.mac %acc, %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<high>} : (!ondsp.acc<storage = i40, frac = 14, signed>, i16, i16) -> !ondsp.acc<storage = i40, frac = 14, signed>
  return %0 : !ondsp.acc<storage = i40, frac = 14, signed>
}

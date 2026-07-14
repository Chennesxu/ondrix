// RUN: not ondrix-opt %s --convert-ondsp-to-ortumcore 2>&1 | FileCheck %s

func.func @q31_mac(%acc: !ondsp.acc<storage = i40, frac = 30, signed>, %a: i32, %b: i32)
    -> !ondsp.acc<storage = i40, frac = 30, signed> {
  // CHECK: q31 high-product target equivalence is not specified; lower through a proven scalar sequence first
  %0 = ondsp.mac %acc, %a, %b {numeric = #ondsp.fixed<signed, storage = i32, frac = 31>, product = #ondsp.product<high>} : (!ondsp.acc<storage = i40, frac = 30, signed>, i32, i32) -> !ondsp.acc<storage = i40, frac = 30, signed>
  return %0 : !ondsp.acc<storage = i40, frac = 30, signed>
}

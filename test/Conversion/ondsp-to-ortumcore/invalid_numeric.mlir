// RUN: not ondrix-opt %s --convert-ondsp-to-ortumcore 2>&1 | FileCheck %s

func.func @unsupported_q7_mac(%acc: !ondsp.acc<storage = i40, frac = 14, signed>,
                              %a: i8, %b: i8)
    -> !ondsp.acc<storage = i40, frac = 14, signed> {
  // CHECK: only signed q15/product=full and signed q31/product=high MAC policies are supported
  %0 = ondsp.mac %acc, %a, %b {numeric = #ondsp.fixed<signed, storage = i8, frac = 7>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 14, signed>, i8, i8) -> !ondsp.acc<storage = i40, frac = 14, signed>
  return %0 : !ondsp.acc<storage = i40, frac = 14, signed>
}

// RUN: not ondrix-opt %s --convert-ondsp-to-ortumcore 2>&1 | FileCheck %s

func.func @q31_full_product_is_not_a_high_product_target(
    %acc: !ondsp.acc<storage = i64, frac = 62, signed>, %a: i32, %b: i32)
    -> !ondsp.acc<storage = i64, frac = 62, signed> {
  // CHECK: signed q31 MAC lowering requires product = #ondsp.product<high>
  %0 = ondsp.mac %acc, %a, %b {numeric = #ondsp.fixed<signed, storage = i32, frac = 31>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i64, frac = 62, signed>, i32, i32) -> !ondsp.acc<storage = i64, frac = 62, signed>
  return %0 : !ondsp.acc<storage = i64, frac = 62, signed>
}

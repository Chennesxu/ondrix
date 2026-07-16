// RUN: not ondrix-opt %s --convert-ondsp-q15-to-scalar 2>&1 | FileCheck %s

func.func @q31_high(
    %acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    %lhs: i32, %rhs: i32)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  // CHECK: error: 'ondsp.mac' op Q15 scalar lowering requires signed i16 frac=15 full product and signed i40 frac=30 accumulator
  %next = ondsp.mac %acc, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i32, i32) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %next : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// RUN: ondrix-opt %s --convert-ondsp-fixed-to-scalar | FileCheck %s

func.func @q15_mac_export(%lhs: i16, %rhs: i16) -> i16 {
  %zero = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %acc = ondsp.mac %zero, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %result : i16
}

// CHECK-LABEL: func.func @q15_mac_export(
// CHECK: arith.constant 0 : i40
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: arith.shrsi
// CHECK: arith.trunci {{.*}} : i40 to i16
// CHECK: return {{.*}} : i16
// CHECK-NOT: ondsp.

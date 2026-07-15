// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s

func.func @dot_quantize(%lhs: i16, %rhs: i16, %wide: i32) -> (i16, i16) {
  %0 = ondrix.dot %lhs, %rhs {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>, tag = "dot"} : (i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %1 = ondsp.acc_export %0 {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, rounding = #ondsp.rounding<nearest_even>, overflow = #ondsp.overflow<saturate>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  %2 = ondrix.quantize %wide {src = #ondsp.fixed<signed, storage = i32, frac = 30>, dst = #ondsp.fixed<signed, storage = i16, frac = 15>, tag = "quantize"} : (i32) -> i16
  return %1, %2 : i16, i16
}

// CHECK-LABEL: func.func @dot_quantize
// CHECK-NOT: ondrix.
// CHECK-NOT: tag =
// CHECK: %[[ZERO:.*]] = ondsp.acc_zero
// CHECK: %[[ACC:.*]] = ondsp.mac %[[ZERO]], %arg0, %arg1
// CHECK: ondsp.acc_export %[[ACC]]
// CHECK: ondsp.convert %arg2 {{.*}}{dst = #ondsp.fixed<signed, storage = i16, frac = 15>, src = #ondsp.fixed<signed, storage = i32, frac = 30>} : (i32) -> i16

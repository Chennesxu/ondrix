// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s

func.func @dot_quantize(%lhs: i16, %rhs: i16) -> i16 {
  %0 = ondrix.dot %lhs, %rhs {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>, tag = "dot"} : (i16, i16) -> i32
  %1 = ondrix.quantize %0 {src = #ondsp.fixed<signed, storage = i32, frac = 30>, dst = #ondsp.fixed<signed, storage = i16, frac = 15>, tag = "quantize"} : (i32) -> i16
  return %1 : i16
}

// CHECK-LABEL: func.func @dot_quantize
// CHECK-NOT: ondrix.
// CHECK-NOT: tag =
// CHECK: ondsp.reduce_mac %arg0, %arg1 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (i16, i16) -> i32
// CHECK: ondsp.convert {{.*}}{dst = #ondsp.fixed<signed, storage = i16, frac = 15>, src = #ondsp.fixed<signed, storage = i32, frac = 30>} : (i32) -> i16

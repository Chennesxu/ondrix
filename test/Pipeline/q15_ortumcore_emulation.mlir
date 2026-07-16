// RUN: ondrix-opt %s --convert-ondsp-to-ortumcore --convert-ortumcore-to-ondsp-emulation | FileCheck %s

func.func @round_trip(%lhs: i16, %rhs: i16)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  %zero = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %result = ondsp.mac %zero, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @round_trip(
// CHECK-SAME: -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
// CHECK: %[[ZERO:.*]] = ondsp.acc_zero
// CHECK: %[[RESULT:.*]] = ondsp.mac %[[ZERO]]
// CHECK: return %[[RESULT]]
// CHECK-NOT: ortumcore.

// RUN: ondrix-opt %s --convert-ondsp-fixed-to-scalar | FileCheck %s

func.func @export_floor_saturate(
    %acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>)
    -> i16 {
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<saturate>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %result : i16
}

func.func @export_zero_wrap(
    %acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>)
    -> i16 {
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_zero>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) -> i16
  return %result : i16
}

func.func @export_nearest_even_wrap(
    %acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>)
    -> i16 {
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %result : i16
}

func.func @export_same_width_wrap(
    %acc: !ondsp.acc<storage = i32, frac = 30, signed, update_overflow = wrap>)
    -> i32 {
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i32, frac = 30>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i32, frac = 30, signed, update_overflow = wrap>) -> i32
  return %result : i32
}

func.func @export_same_width_saturate(
    %acc: !ondsp.acc<storage = i32, frac = 30, signed, update_overflow = saturate>)
    -> i32 {
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i32, frac = 30>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (!ondsp.acc<storage = i32, frac = 30, signed, update_overflow = saturate>) -> i32
  return %result : i32
}

// CHECK-LABEL: func.func @export_floor_saturate(
// CHECK-SAME: %[[ACC:.*]]: i40) -> i16
// CHECK: %[[SHIFT:.*]] = arith.constant 15 : i40
// CHECK: %[[ROUNDED:.*]] = arith.shrsi %[[ACC]], %[[SHIFT]] : i40
// CHECK: %[[MIN:.*]] = arith.constant -32768 : i40
// CHECK: %[[MAX:.*]] = arith.constant 32767 : i40
// CHECK: %[[BELOW:.*]] = arith.cmpi slt, %[[ROUNDED]], %[[MIN]] : i40
// CHECK: %[[ABOVE:.*]] = arith.cmpi sgt, %[[ROUNDED]], %[[MAX]] : i40
// CHECK: %[[LOWER:.*]] = arith.select %[[BELOW]], %[[MIN]], %[[ROUNDED]] : i40
// CHECK: %[[CLAMPED:.*]] = arith.select %[[ABOVE]], %[[MAX]], %[[LOWER]] : i40
// CHECK: %[[RESULT:.*]] = arith.trunci %[[CLAMPED]] : i40 to i16
// CHECK: return %[[RESULT]] : i16

// CHECK-LABEL: func.func @export_zero_wrap(
// CHECK-SAME: %[[ACC:.*]]: i40) -> i16
// CHECK: %[[SHIFT:.*]] = arith.constant 15 : i40
// CHECK: %[[QUOTIENT:.*]] = arith.shrsi %[[ACC]], %[[SHIFT]] : i40
// CHECK: %[[BITS:.*]] = arith.trunci %[[ACC]] : i40 to i15
// CHECK: %[[REMAINDER:.*]] = arith.extui %[[BITS]] : i15 to i40
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i40
// CHECK: %[[ONE:.*]] = arith.constant 1 : i40
// CHECK: %[[NEGATIVE:.*]] = arith.cmpi slt, %[[ACC]], %[[ZERO]] : i40
// CHECK: %[[NONZERO:.*]] = arith.cmpi ne, %[[REMAINDER]], %[[ZERO]] : i40
// CHECK: %[[INCREMENT_IF:.*]] = arith.andi %[[NEGATIVE]], %[[NONZERO]] : i1
// CHECK: %[[INCREMENT:.*]] = arith.select %[[INCREMENT_IF]], %[[ONE]], %[[ZERO]] : i40
// CHECK: %[[ROUNDED:.*]] = arith.addi %[[QUOTIENT]], %[[INCREMENT]] : i40
// CHECK: %[[RESULT:.*]] = arith.trunci %[[ROUNDED]] : i40 to i16
// CHECK: return %[[RESULT]] : i16

// CHECK-LABEL: func.func @export_nearest_even_wrap(
// CHECK-SAME: %[[ACC:.*]]: i40) -> i16
// CHECK: %[[QUOTIENT:.*]] = arith.shrsi %[[ACC]], %{{.*}} : i40
// CHECK: %[[BITS:.*]] = arith.trunci %[[ACC]] : i40 to i15
// CHECK: %[[REMAINDER:.*]] = arith.extui %[[BITS]] : i15 to i40
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i40
// CHECK: %[[ONE:.*]] = arith.constant 1 : i40
// CHECK: %[[HALF:.*]] = arith.constant 16384 : i40
// CHECK: %[[ABOVE:.*]] = arith.cmpi ugt, %[[REMAINDER]], %[[HALF]] : i40
// CHECK: %[[EQUAL:.*]] = arith.cmpi eq, %[[REMAINDER]], %[[HALF]] : i40
// CHECK: %[[LOW_BIT:.*]] = arith.andi %[[QUOTIENT]], %[[ONE]] : i40
// CHECK: %[[ODD:.*]] = arith.cmpi ne, %[[LOW_BIT]], %[[ZERO]] : i40
// CHECK: %[[TIE_ODD:.*]] = arith.andi %[[EQUAL]], %[[ODD]] : i1
// CHECK: %[[INCREMENT_IF:.*]] = arith.ori %[[ABOVE]], %[[TIE_ODD]] : i1
// CHECK: %[[INCREMENT:.*]] = arith.select %[[INCREMENT_IF]], %[[ONE]], %[[ZERO]] : i40
// CHECK: %[[ROUNDED:.*]] = arith.addi %[[QUOTIENT]], %[[INCREMENT]] : i40
// CHECK: %[[RESULT:.*]] = arith.trunci %[[ROUNDED]] : i40 to i16
// CHECK: return %[[RESULT]] : i16

// CHECK-LABEL: func.func @export_same_width_wrap(
// CHECK-SAME: %[[ACC:.*]]: i32) -> i32
// CHECK-NEXT: return %[[ACC]] : i32

// CHECK-LABEL: func.func @export_same_width_saturate(
// CHECK-SAME: %[[ACC:.*]]: i32) -> i32
// CHECK-NEXT: return %[[ACC]] : i32
// CHECK-NOT: ondsp.

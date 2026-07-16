// RUN: ondrix-opt %s --convert-ondsp-fixed-to-scalar | FileCheck %s

func.func @import_q31(%input: i32)
    -> !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate> {
  %acc = ondsp.acc_import %input {
    src = #ondsp.fixed<signed, storage = i32, frac = 31>
  } : (i32) -> !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>
  return %acc : !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>
}

func.func @import_q30(%input: i32)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap> {
  %acc = ondsp.acc_import %input {
    src = #ondsp.fixed<signed, storage = i32, frac = 30>
  } : (i32) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  return %acc : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
}

func.func @export_q31(
    %acc: !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>)
    -> i32 {
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i32, frac = 31>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>) -> i32
  return %result : i32
}

func.func @export_q30(
    %acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>)
    -> i32 {
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i32, frac = 30>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) -> i32
  return %result : i32
}

// CHECK-LABEL: func.func @import_q31(
// CHECK-SAME: %[[INPUT:.*]]: i32) -> i64
// CHECK: %[[EXTENDED:.*]] = arith.extsi %[[INPUT]] : i32 to i64
// CHECK: %[[SHIFT:.*]] = arith.constant 31 : i64
// CHECK: %[[ACC:.*]] = arith.shli %[[EXTENDED]], %[[SHIFT]] : i64
// CHECK: return %[[ACC]] : i64

// CHECK-LABEL: func.func @import_q30(
// CHECK-SAME: %[[INPUT:.*]]: i32) -> i40
// CHECK: %[[EXTENDED:.*]] = arith.extsi %[[INPUT]] : i32 to i40
// CHECK: %[[SHIFT:.*]] = arith.constant 0 : i40
// CHECK: %[[ACC:.*]] = arith.shli %[[EXTENDED]], %[[SHIFT]] : i40
// CHECK: return %[[ACC]] : i40

// CHECK-LABEL: func.func @export_q31(
// CHECK-SAME: %[[ACC:.*]]: i64) -> i32
// CHECK: %[[SHIFT:.*]] = arith.constant 31 : i64
// CHECK: %[[QUOTIENT:.*]] = arith.shrsi %[[ACC]], %[[SHIFT]] : i64
// CHECK: %[[BITS:.*]] = arith.trunci %[[ACC]] : i64 to i31
// CHECK: %[[REMAINDER:.*]] = arith.extui %[[BITS]] : i31 to i64
// CHECK: %[[HALF:.*]] = arith.constant 1073741824 : i64
// CHECK: %[[RESULT:.*]] = arith.trunci {{.*}} : i64 to i32
// CHECK: return %[[RESULT]] : i32

// CHECK-LABEL: func.func @export_q30(
// CHECK-SAME: %[[ACC:.*]]: i40) -> i32
// CHECK-NEXT: %[[RESULT:.*]] = arith.trunci %[[ACC]] : i40 to i32
// CHECK-NEXT: return %[[RESULT]] : i32
// CHECK-NOT: ondsp.

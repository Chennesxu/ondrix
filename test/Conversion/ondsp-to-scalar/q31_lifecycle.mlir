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
// CHECK-SAME: %[[INPUT:.*]]: i32) -> i64
// CHECK: %[[EXTENDED:.*]] = arith.extsi %[[INPUT]] : i32 to i64
// CHECK: %[[SHIFT:.*]] = arith.constant 0 : i64
// CHECK: %[[ACC:.*]] = arith.shli %[[EXTENDED]], %[[SHIFT]] : i64
// CHECK: return %[[ACC]] : i64

// CHECK-LABEL: func.func @export_q31(
// CHECK-SAME: %[[ACC:.*]]: i64) -> i32
// CHECK: %[[SHIFT:.*]] = arith.constant 31 : i64
// CHECK: %[[QUOTIENT:.*]] = arith.shrsi %[[ACC]], %[[SHIFT]] : i64
// CHECK: %[[WINDOW:.*]] = arith.trunci %[[ACC]] : i64 to i32
// CHECK: %[[MASK:.*]] = arith.constant 2147483647 : i32
// CHECK: %[[REMAINDER:.*]] = arith.andi %[[WINDOW]], %[[MASK]] : i32
// CHECK: %[[LOW_BIT:.*]] = arith.andi %{{.*}}, {{.*}} : i32
// CHECK: %[[SUM:.*]] = arith.addi %[[REMAINDER]], %[[LOW_BIT]] : i32
// CHECK: %[[BELOW_HALF:.*]] = arith.constant 1073741823 : i32
// CHECK: %[[BIASED:.*]] = arith.addi %[[SUM]], %[[BELOW_HALF]] : i32
// CHECK: %[[CARRY:.*]] = arith.shrui %[[BIASED]], {{.*}} : i32
// CHECK: %[[WIDE_CARRY:.*]] = arith.extui %[[CARRY]] : i32 to i64
// CHECK: %[[ROUNDED:.*]] = arith.addi %[[QUOTIENT]], %[[WIDE_CARRY]] : i64
// CHECK: %[[NARROWED:.*]] = arith.trunci %[[ROUNDED]] : i64 to i32
// CHECK: %[[FITS:.*]] = arith.cmpi eq, %{{.*}}, %[[ROUNDED]] : i64
// CHECK: %[[RESULT:.*]] = arith.select %[[FITS]], %[[NARROWED]], %{{.*}} : i32
// CHECK: return %[[RESULT]] : i32

// CHECK-LABEL: func.func @export_q30(
// CHECK-SAME: %[[ACC:.*]]: i64) -> i32
// CHECK-NEXT: %[[NARROW:.*]] = arith.trunci %[[ACC]] : i64 to i40
// CHECK-NEXT: %[[WRAPPED:.*]] = arith.extsi %[[NARROW]] : i40 to i64
// CHECK-NEXT: %[[RESULT:.*]] = arith.trunci %[[WRAPPED]] : i64 to i32
// CHECK-NEXT: return %[[RESULT]] : i32
// CHECK-NOT: ondsp.

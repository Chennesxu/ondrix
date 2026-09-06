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

func.func @export_ties_positive_saturate(
    %acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>)
    -> i16 {
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_ties_positive>,
    overflow = #ondsp.overflow<saturate>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) -> i16
  return %result : i16
}

func.func @export_ties_positive_full_width(
    %acc: !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>)
    -> i32 {
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i32, frac = 31>,
    rounding = #ondsp.rounding<nearest_ties_positive>,
    overflow = #ondsp.overflow<saturate>
  } : (!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>) -> i32
  return %result : i32
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
// CHECK-SAME: %[[ACC:.*]]: i64) -> i16
// CHECK: %[[SHIFT:.*]] = arith.constant 15 : i64
// CHECK: %[[ROUNDED:.*]] = arith.shrsi %[[ACC]], %[[SHIFT]] : i64
// CHECK: %[[MIN:.*]] = arith.constant -32768 : i64
// CHECK: %[[MAX:.*]] = arith.constant 32767 : i64
// CHECK: %[[LOWER:.*]] = arith.maxsi %[[ROUNDED]], %[[MIN]] : i64
// CHECK: %[[CLAMPED:.*]] = arith.minsi %[[LOWER]], %[[MAX]] : i64
// CHECK: %[[RESULT:.*]] = arith.trunci %[[CLAMPED]] : i64 to i16
// CHECK: return %[[RESULT]] : i16

// CHECK-LABEL: func.func @export_zero_wrap(
// CHECK-SAME: %[[ACC:.*]]: i64) -> i16
// CHECK: %[[NARROW:.*]] = arith.trunci %[[ACC]] : i64 to i40
// CHECK: %[[WRAPPED:.*]] = arith.extsi %[[NARROW]] : i40 to i64
// CHECK: %[[SHIFT:.*]] = arith.constant 15 : i64
// CHECK: %[[QUOTIENT:.*]] = arith.shrsi %[[WRAPPED]], %[[SHIFT]] : i64
// CHECK: %[[BITS:.*]] = arith.trunci %[[WRAPPED]] : i64 to i15
// CHECK: %[[REMAINDER:.*]] = arith.extui %[[BITS]] : i15 to i64
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i64
// CHECK: %[[ONE:.*]] = arith.constant 1 : i64
// CHECK: %[[NEGATIVE:.*]] = arith.cmpi slt, %[[WRAPPED]], %[[ZERO]] : i64
// CHECK: %[[NONZERO:.*]] = arith.cmpi ne, %[[REMAINDER]], %[[ZERO]] : i64
// CHECK: %[[INCREMENT_IF:.*]] = arith.andi %[[NEGATIVE]], %[[NONZERO]] : i1
// CHECK: %[[INCREMENT:.*]] = arith.select %[[INCREMENT_IF]], %[[ONE]], %[[ZERO]] : i64
// CHECK: %[[ROUNDED:.*]] = arith.addi %[[QUOTIENT]], %[[INCREMENT]] : i64
// CHECK: %[[RESULT:.*]] = arith.trunci %[[ROUNDED]] : i64 to i16
// CHECK: return %[[RESULT]] : i16

// CHECK-LABEL: func.func @export_nearest_even_wrap(
// CHECK-SAME: %[[ACC:.*]]: i64) -> i16
// CHECK: %[[QUOTIENT:.*]] = arith.shrsi %[[ACC]], %{{.*}} : i64
// CHECK: %[[BITS:.*]] = arith.trunci %[[ACC]] : i64 to i15
// CHECK: %[[REMAINDER:.*]] = arith.extui %[[BITS]] : i15 to i64
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i64
// CHECK: %[[ONE:.*]] = arith.constant 1 : i64
// CHECK: %[[HALF:.*]] = arith.constant 16384 : i64
// CHECK: %[[ABOVE:.*]] = arith.cmpi ugt, %[[REMAINDER]], %[[HALF]] : i64
// CHECK: %[[TIE_MASK:.*]] = arith.constant 65535 : i64
// CHECK: %[[TIE_BITS:.*]] = arith.andi %[[ACC]], %[[TIE_MASK]] : i64
// CHECK: %[[TIE_ODD_PATTERN:.*]] = arith.constant 49152 : i64
// CHECK: %[[TIE_ODD:.*]] = arith.cmpi eq, %[[TIE_BITS]], %[[TIE_ODD_PATTERN]] : i64
// CHECK: %[[INCREMENT_IF:.*]] = arith.ori %[[ABOVE]], %[[TIE_ODD]] : i1
// CHECK: %[[INCREMENT:.*]] = arith.select %[[INCREMENT_IF]], %[[ONE]], %[[ZERO]] : i64
// CHECK: %[[ROUNDED:.*]] = arith.addi %[[QUOTIENT]], %[[INCREMENT]] : i64
// CHECK: %[[RESULT:.*]] = arith.trunci %[[ROUNDED]] : i64 to i16
// CHECK: return %[[RESULT]] : i16

// The i40 value inside the i64 carrier has headroom for the add-half, so
// ties-positive is one add and one shift; no remainder compare.
// CHECK-LABEL: func.func @export_ties_positive_saturate(
// CHECK-SAME: %[[ACC:.*]]: i64) -> i16
// CHECK: %[[NARROW:.*]] = arith.trunci %[[ACC]] : i64 to i40
// CHECK: %[[WRAPPED:.*]] = arith.extsi %[[NARROW]] : i40 to i64
// CHECK: %[[HALF:.*]] = arith.constant 16384 : i64
// CHECK: %[[BIASED:.*]] = arith.addi %[[WRAPPED]], %[[HALF]] : i64
// CHECK: %[[SHIFT:.*]] = arith.constant 15 : i64
// CHECK: %[[ROUNDED:.*]] = arith.shrsi %[[BIASED]], %[[SHIFT]] : i64
// CHECK-NOT: arith.cmpi
// CHECK: %[[LOWER:.*]] = arith.maxsi %[[ROUNDED]], %{{.*}} : i64
// CHECK: %[[CLAMPED:.*]] = arith.minsi %[[LOWER]], %{{.*}} : i64
// CHECK: %[[RESULT:.*]] = arith.trunci %[[CLAMPED]] : i64 to i16
// CHECK: return %[[RESULT]] : i16

// A storage as wide as its carrier has no headroom: the remainder form stays.
// CHECK-LABEL: func.func @export_ties_positive_full_width(
// CHECK-SAME: %[[ACC:.*]]: i64) -> i32
// CHECK: %[[QUOTIENT:.*]] = arith.shrsi %[[ACC]], %{{.*}} : i64
// CHECK: arith.cmpi uge
// CHECK: %[[ROUNDED:.*]] = arith.addi %[[QUOTIENT]], %{{.*}} : i64
// CHECK: return

// CHECK-LABEL: func.func @export_same_width_wrap(
// CHECK-SAME: %[[ACC:.*]]: i32) -> i32
// CHECK-NEXT: return %[[ACC]] : i32

// CHECK-LABEL: func.func @export_same_width_saturate(
// CHECK-SAME: %[[ACC:.*]]: i32) -> i32
// CHECK-NEXT: return %[[ACC]] : i32
// CHECK-NOT: ondsp.

// RUN: ondrix-opt %s --convert-ondsp-fixed-to-scalar | FileCheck %s

// A signed i32 destination lowers at any fractional position the verifier
// admits, not only at frac 30. Every export is a value-preserving format
// conversion: the shift is `acc.frac - dst.frac` and the destination frac is
// the reading of the result. Arithmetic scalings that change the value
// (such as a mean by a power of two) belong to `round_shift`, not here.

func.func @export_mean_nearest_even_saturate(
    %acc: !ondsp.acc<storage = i64, frac = 30, signed, update_overflow = wrap>)
    -> i32 {
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i32, frac = 24>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (!ondsp.acc<storage = i64, frac = 30, signed, update_overflow = wrap>) -> i32
  return %result : i32
}

func.func @export_mean_floor_saturate(
    %acc: !ondsp.acc<storage = i64, frac = 30, signed, update_overflow = wrap>)
    -> i32 {
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i32, frac = 29>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<saturate>
  } : (!ondsp.acc<storage = i64, frac = 30, signed, update_overflow = wrap>) -> i32
  return %result : i32
}

// Dividing by 2^(30 - 24) = 2^6 with a nearest-even tie at 2^5, then
// saturating into the i32 destination range. The tie test is the carry out of
// the six-bit remainder window, so the constants pin the shift and the tie.
// CHECK-LABEL: func.func @export_mean_nearest_even_saturate(
// CHECK-SAME: %[[ACC:.*]]: i64) -> i32
// CHECK: %[[SHIFT:.*]] = arith.constant 6 : i64
// CHECK: %[[QUOTIENT:.*]] = arith.shrsi %[[ACC]], %[[SHIFT]] : i64
// CHECK: %[[MASK:.*]] = arith.constant 63 : i64
// CHECK: %[[REMAINDER:.*]] = arith.andi %[[ACC]], %[[MASK]] : i64
// CHECK: %[[ONE:.*]] = arith.constant 1 : i64
// CHECK: %[[LOW_BIT:.*]] = arith.andi %[[QUOTIENT]], %[[ONE]] : i64
// CHECK: %[[SUM:.*]] = arith.addi %[[REMAINDER]], %[[LOW_BIT]] : i64
// CHECK: %[[BELOW_HALF:.*]] = arith.constant 31 : i64
// CHECK: %[[BIASED:.*]] = arith.addi %[[SUM]], %[[BELOW_HALF]] : i64
// CHECK: %[[CARRY:.*]] = arith.shrui %[[BIASED]], %{{.*}} : i64
// CHECK: %[[ROUNDED:.*]] = arith.addi %[[QUOTIENT]], %[[CARRY]] : i64
// CHECK-NOT: arith.select
// CHECK: %[[NARROWED:.*]] = arith.trunci %[[ROUNDED]] : i64 to i32
// CHECK: %[[WIDENED:.*]] = arith.extsi %[[NARROWED]] : i32 to i64
// CHECK: %[[FITS:.*]] = arith.cmpi eq, %[[WIDENED]], %[[ROUNDED]] : i64
// CHECK: %[[MAX:.*]] = arith.constant 2147483647 : i32
// CHECK: %[[RAIL:.*]] = arith.addi %{{.*}}, %[[MAX]] : i32
// CHECK: %[[RESULT:.*]] = arith.select %[[FITS]], %[[NARROWED]], %[[RAIL]] : i32
// CHECK: return %[[RESULT]] : i32

// Floor rounding at frac 29 is one arithmetic shift by 2^(30 - 29).
// CHECK-LABEL: func.func @export_mean_floor_saturate(
// CHECK-SAME: %[[FLOOR_ACC:.*]]: i64) -> i32
// CHECK: %[[FLOOR_SHIFT:.*]] = arith.constant 1 : i64
// CHECK: %[[FLOOR_ROUNDED:.*]] = arith.shrsi %[[FLOOR_ACC]], %[[FLOOR_SHIFT]] : i64
// CHECK: %[[FLOOR_NARROWED:.*]] = arith.trunci %[[FLOOR_ROUNDED]] : i64 to i32
// CHECK: %[[FLOOR_FITS:.*]] = arith.cmpi eq, %{{.*}}, %[[FLOOR_ROUNDED]] : i64
// CHECK: %[[FLOOR_RESULT:.*]] = arith.select %[[FLOOR_FITS]], %[[FLOOR_NARROWED]], %{{.*}} : i32
// CHECK: return %[[FLOOR_RESULT]] : i32
// CHECK-NOT: ondsp.

// The identity signed i64/frac30 destination materializes the raw
// accumulator value unchanged: zero shift, same storage, no clamp.
// CHECK-LABEL: func.func @export_identity_sum(
// CHECK-SAME: %[[IDENTITY_ACC:.*]]: i64) -> i64
// CHECK-NEXT: return %[[IDENTITY_ACC]] : i64
func.func @export_identity_sum(
    %acc: !ondsp.acc<storage = i64, frac = 30, signed, update_overflow = wrap>)
    -> i64 {
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i64, frac = 30>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (!ondsp.acc<storage = i64, frac = 30, signed, update_overflow = wrap>) -> i64
  return %result : i64
}

// The frac 0 endpoint of the widened domain: the full 30-position shift, whose
// remainder window carries the nearest-even bias 2^29 - 1.
// CHECK-LABEL: func.func @export_integer_reading(
// CHECK: arith.constant 30 : i64
// CHECK: arith.shrsi
// CHECK: arith.constant 536870911 : i64
// CHECK: arith.trunci {{.*}} : i64 to i32
func.func @export_integer_reading(
    %acc: !ondsp.acc<storage = i64, frac = 30, signed, update_overflow = wrap>)
    -> i32 {
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i32, frac = 0>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (!ondsp.acc<storage = i64, frac = 30, signed, update_overflow = wrap>) -> i32
  return %result : i32
}

// A narrower frac30 accumulator shares the i64 carrier, so the identity
// i64/frac30 destination only re-wraps to the storage width, and never clamps.
// CHECK-LABEL: func.func @export_widen_i40_wrap(
// CHECK-SAME: %[[W40:.*]]: i64) -> i64
// CHECK-NEXT: %[[W40_NARROW:.*]] = arith.trunci %[[W40]] : i64 to i40
// CHECK-NEXT: %[[W40_RESULT:.*]] = arith.extsi %[[W40_NARROW]] : i40 to i64
// CHECK-NEXT: return %[[W40_RESULT]] : i64
// CHECK-NOT: arith.maxsi
func.func @export_widen_i40_wrap(
    %acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>)
    -> i64 {
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i64, frac = 30>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) -> i64
  return %result : i64
}

// CHECK-LABEL: func.func @export_widen_i40_saturate(
// CHECK-SAME: %[[S40:.*]]: i64) -> i64
// CHECK-NEXT: return %[[S40]] : i64
func.func @export_widen_i40_saturate(
    %acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>)
    -> i64 {
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i64, frac = 30>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i64
  return %result : i64
}

// CHECK-LABEL: func.func @export_widen_i48_wrap(
// CHECK-SAME: %[[W48:.*]]: i64) -> i64
// CHECK-NEXT: %[[W48_NARROW:.*]] = arith.trunci %[[W48]] : i64 to i48
// CHECK-NEXT: %[[W48_RESULT:.*]] = arith.extsi %[[W48_NARROW]] : i48 to i64
// CHECK-NEXT: return %[[W48_RESULT]] : i64
// CHECK-NOT: arith.maxsi
func.func @export_widen_i48_wrap(
    %acc: !ondsp.acc<storage = i48, frac = 30, signed, update_overflow = wrap>)
    -> i64 {
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i64, frac = 30>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i48, frac = 30, signed, update_overflow = wrap>) -> i64
  return %result : i64
}

// CHECK-LABEL: func.func @export_widen_i48_saturate(
// CHECK-SAME: %[[S48:.*]]: i64) -> i64
// CHECK-NEXT: return %[[S48]] : i64
func.func @export_widen_i48_saturate(
    %acc: !ondsp.acc<storage = i48, frac = 30, signed, update_overflow = saturate>)
    -> i64 {
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i64, frac = 30>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (!ondsp.acc<storage = i48, frac = 30, signed, update_overflow = saturate>) -> i64
  return %result : i64
}

// CHECK-NOT: ondsp.

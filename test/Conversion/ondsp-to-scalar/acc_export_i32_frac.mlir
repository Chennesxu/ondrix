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
// saturating into the i32 destination range.
// CHECK-LABEL: func.func @export_mean_nearest_even_saturate(
// CHECK-SAME: %[[ACC:.*]]: i64) -> i32
// CHECK: %[[SHIFT:.*]] = arith.constant 6 : i64
// CHECK: %[[QUOTIENT:.*]] = arith.shrsi %[[ACC]], %[[SHIFT]] : i64
// CHECK: %[[BITS:.*]] = arith.trunci %[[ACC]] : i64 to i6
// CHECK: %[[REMAINDER:.*]] = arith.extui %[[BITS]] : i6 to i64
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i64
// CHECK: %[[ONE:.*]] = arith.constant 1 : i64
// CHECK: %[[HALF:.*]] = arith.constant 32 : i64
// CHECK: %[[ABOVE:.*]] = arith.cmpi ugt, %[[REMAINDER]], %[[HALF]] : i64
// CHECK: %[[EQUAL:.*]] = arith.cmpi eq, %[[REMAINDER]], %[[HALF]] : i64
// CHECK: %[[LOW_BIT:.*]] = arith.andi %[[QUOTIENT]], %[[ONE]] : i64
// CHECK: %[[ODD:.*]] = arith.cmpi ne, %[[LOW_BIT]], %[[ZERO]] : i64
// CHECK: %[[TIE_ODD:.*]] = arith.andi %[[EQUAL]], %[[ODD]] : i1
// CHECK: %[[INCREMENT_IF:.*]] = arith.ori %[[ABOVE]], %[[TIE_ODD]] : i1
// CHECK: %[[INCREMENT:.*]] = arith.select %[[INCREMENT_IF]], %[[ONE]], %[[ZERO]] : i64
// CHECK: %[[ROUNDED:.*]] = arith.addi %[[QUOTIENT]], %[[INCREMENT]] : i64
// CHECK: %[[MIN:.*]] = arith.constant -2147483648 : i64
// CHECK: %[[MAX:.*]] = arith.constant 2147483647 : i64
// CHECK: %[[BELOW:.*]] = arith.cmpi slt, %[[ROUNDED]], %[[MIN]] : i64
// CHECK: %[[OVER:.*]] = arith.cmpi sgt, %[[ROUNDED]], %[[MAX]] : i64
// CHECK: %[[LOWER:.*]] = arith.select %[[BELOW]], %[[MIN]], %[[ROUNDED]] : i64
// CHECK: %[[CLAMPED:.*]] = arith.select %[[OVER]], %[[MAX]], %[[LOWER]] : i64
// CHECK: %[[RESULT:.*]] = arith.trunci %[[CLAMPED]] : i64 to i32
// CHECK: return %[[RESULT]] : i32

// Floor rounding at frac 29 is one arithmetic shift by 2^(30 - 29).
// CHECK-LABEL: func.func @export_mean_floor_saturate(
// CHECK-SAME: %[[FLOOR_ACC:.*]]: i64) -> i32
// CHECK: %[[FLOOR_SHIFT:.*]] = arith.constant 1 : i64
// CHECK: %[[FLOOR_ROUNDED:.*]] = arith.shrsi %[[FLOOR_ACC]], %[[FLOOR_SHIFT]] : i64
// CHECK: arith.select
// CHECK: %[[FLOOR_CLAMPED:.*]] = arith.select
// CHECK: %[[FLOOR_RESULT:.*]] = arith.trunci %[[FLOOR_CLAMPED]] : i64 to i32
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

// The frac 0 endpoint of the widened domain: the full 30-position shift with
// the nearest-even half constant 2^29.
// CHECK-LABEL: func.func @export_integer_reading(
// CHECK: arith.constant 30 : i64
// CHECK: arith.shrsi
// CHECK: arith.constant 536870912 : i64
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

// The identity i64/frac30 destination is also reachable from a NARROWER
// frac30 accumulator. The shift is zero and the frac is unchanged, so the
// only work is a widening sign extension, which is exactly value
// preserving: both declared overflow modes are no-ops and neither a
// truncation nor a clamp may appear.
// CHECK-LABEL: func.func @export_widen_i40_wrap(
// CHECK-SAME: %[[W40:.*]]: i40) -> i64
// CHECK-NEXT: %[[W40_RESULT:.*]] = arith.extsi %[[W40]] : i40 to i64
// CHECK-NEXT: return %[[W40_RESULT]] : i64
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
// CHECK-SAME: %[[S40:.*]]: i40) -> i64
// CHECK-NEXT: %[[S40_RESULT:.*]] = arith.extsi %[[S40]] : i40 to i64
// CHECK-NEXT: return %[[S40_RESULT]] : i64
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
// CHECK-SAME: %[[W48:.*]]: i48) -> i64
// CHECK-NEXT: %[[W48_RESULT:.*]] = arith.extsi %[[W48]] : i48 to i64
// CHECK-NEXT: return %[[W48_RESULT]] : i64
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
// CHECK-SAME: %[[S48:.*]]: i48) -> i64
// CHECK-NEXT: %[[S48_RESULT:.*]] = arith.extsi %[[S48]] : i48 to i64
// CHECK-NEXT: return %[[S48_RESULT]] : i64
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

// RUN: ondrix-opt %s --convert-ondsp-fixed-to-scalar | FileCheck %s

// CHECK-LABEL: func.func @round_shift_floor_keeps_width
// CHECK: %[[SHIFT:.*]] = arith.constant 16 : i32
// CHECK: %[[Q:.*]] = arith.shrsi %{{.*}}, %[[SHIFT]] : i32
// CHECK-NOT: arith.select
// CHECK: return %[[Q]] : i32
func.func @round_shift_floor_keeps_width(%input: i32) -> i32 {
  %0 = ondsp.round_shift %input {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 16, rounding = toward_negative, overflow = wrap, saturate_to = i32>} : (i32) -> i32
  return %0 : i32
}

// Nearest-even is the carry out of the 15-bit remainder window: the quotient's
// low bit joins the remainder and 2^14 - 1 decides the tie, so no compare pair
// and no select survive.
// CHECK-LABEL: func.func @round_shift_nearest_even_saturating_narrow
// CHECK: %[[Q:.*]] = arith.shrsi
// CHECK: %[[REM:.*]] = arith.andi %{{.*}}, %{{.*}} : i32
// CHECK: %[[LOW_BIT:.*]] = arith.andi %[[Q]], %{{.*}} : i32
// CHECK: %[[SUM:.*]] = arith.addi %[[REM]], %[[LOW_BIT]] : i32
// CHECK: %[[BELOW_HALF:.*]] = arith.constant 16383 : i32
// CHECK: %[[BIASED:.*]] = arith.addi %[[SUM]], %[[BELOW_HALF]] : i32
// CHECK: %[[CARRY:.*]] = arith.shrui %[[BIASED]], %{{.*}} : i32
// CHECK: arith.addi %[[Q]], %[[CARRY]] : i32
// CHECK-NOT: arith.cmpi
// CHECK-NOT: arith.select
// CHECK: arith.trunci %{{.*}} : i32 to i16
func.func @round_shift_nearest_even_saturating_narrow(%input: i32) -> i16 {
  %0 = ondsp.round_shift %input {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>} : (i32) -> i16
  return %0 : i16
}

// Ties toward positive infinity: one unsigned `uge` test against half decides
// the increment. The quotient parity that nearest_even inspects is absent,
// and so is any in-width add of half (which would overflow near the maximum).
// CHECK-LABEL: func.func @round_shift_nearest_ties_positive_saturating_narrow
// CHECK: %[[SHIFT:.*]] = arith.constant 15 : i32
// CHECK: %[[Q:.*]] = arith.shrsi %{{.*}}, %[[SHIFT]] : i32
// CHECK: %[[BITS:.*]] = arith.trunci %{{.*}} : i32 to i15
// CHECK: %[[REM:.*]] = arith.extui %[[BITS]] : i15 to i32
// CHECK: %[[HALF:.*]] = arith.constant 16384 : i32
// CHECK: %[[TIE:.*]] = arith.cmpi uge, %[[REM]], %[[HALF]] : i32
// CHECK: %[[INC:.*]] = arith.select %[[TIE]], %{{.*}}, %{{.*}} : i32
// CHECK: arith.addi %[[Q]], %[[INC]] : i32
// CHECK-NOT: arith.andi
// CHECK-NOT: arith.ori
// CHECK-NOT: arith.cmpi eq
// CHECK: arith.trunci %{{.*}} : i32 to i16
func.func @round_shift_nearest_ties_positive_saturating_narrow(%input: i32) -> i16 {
  %0 = ondsp.round_shift %input {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_ties_positive, overflow = saturate, saturate_to = i16>} : (i32) -> i16
  return %0 : i16
}

// CHECK-LABEL: func.func @sat_cast_clamps_to_narrow_storage
// CHECK: %[[NARROWED:.*]] = arith.trunci %{{.*}} : i32 to i16
// CHECK: %[[FITS:.*]] = arith.cmpi eq, %{{.*}} : i32
// CHECK: arith.constant 32767 : i16
// CHECK: arith.select %[[FITS]], %[[NARROWED]], %{{.*}} : i16
func.func @sat_cast_clamps_to_narrow_storage(%input: i32) -> i16 {
  %0 = ondsp.sat_cast %input {numeric = #ondsp.fixed<signed, storage = i16, frac = 12>} : (i32) -> i16
  return %0 : i16
}

// CHECK-LABEL: func.func @sat_cast_widens_exactly
// CHECK: arith.extsi %{{.*}} : i16 to i32
// CHECK-NOT: arith.select
func.func @sat_cast_widens_exactly(%input: i16) -> i32 {
  %0 = ondsp.sat_cast %input {numeric = #ondsp.fixed<signed, storage = i32, frac = 30>} : (i16) -> i32
  return %0 : i32
}

// CHECK-LABEL: func.func @round_shift_floor_fixed_vector
// CHECK: arith.shrsi %{{.*}} : vector<4xi32>
// CHECK-NOT: ondsp.
func.func @round_shift_floor_fixed_vector(%input: vector<4xi32>) -> vector<4xi32> {
  %0 = ondsp.round_shift %input {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = wrap, saturate_to = i32>} : (vector<4xi32>) -> vector<4xi32>
  return %0 : vector<4xi32>
}

// CHECK-LABEL: func.func @sat_cast_clamps_fixed_vector
// CHECK: arith.maxsi %{{.*}} : vector<4xi32>
// CHECK: arith.minsi %{{.*}} : vector<4xi32>
// CHECK: arith.trunci %{{.*}} : vector<4xi32> to vector<4xi16>
func.func @sat_cast_clamps_fixed_vector(%input: vector<4xi32>) -> vector<4xi16> {
  %0 = ondsp.sat_cast %input {numeric = #ondsp.fixed<signed, storage = i16, frac = 12>} : (vector<4xi32>) -> vector<4xi16>
  return %0 : vector<4xi16>
}

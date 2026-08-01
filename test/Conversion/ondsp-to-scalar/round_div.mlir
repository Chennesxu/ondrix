// RUN: ondrix-opt %s --convert-ondsp-fixed-to-scalar | FileCheck %s

// The lowering recovers the contract's Euclidean pair from arith's
// toward-zero division, then decides the increment from the remainder
// against divisor - remainder: no doubled remainder and no add-half exists
// anywhere. The final clamp is the declared saturating narrowing.

// CHECK-LABEL: func.func @round_div_nearest_even
// CHECK: %[[DIV:.*]] = arith.divsi %{{.*}}, %[[D:.*]] : i64
// CHECK: %[[REM:.*]] = arith.remsi %{{.*}}, %[[D]] : i64
// CHECK: %[[NEG:.*]] = arith.cmpi slt, %[[REM]], %{{.*}} : i64
// CHECK: %[[QSTEP:.*]] = arith.subi %[[DIV]], %{{.*}} : i64
// CHECK: %[[Q:.*]] = arith.select %[[NEG]], %[[QSTEP]], %[[DIV]] : i64
// CHECK: %[[RLIFT:.*]] = arith.addi %[[REM]], %[[D]] : i64
// CHECK: %[[R:.*]] = arith.select %[[NEG]], %[[RLIFT]], %[[REM]] : i64
// CHECK: %[[COMP:.*]] = arith.subi %[[D]], %[[R]] : i64
// CHECK: arith.cmpi sgt, %[[R]], %[[COMP]] : i64
// CHECK: arith.cmpi eq, %[[R]], %[[COMP]] : i64
// CHECK: arith.trunci {{.*}} : i64 to i16
// CHECK-NOT: ondsp.round_div
func.func @round_div_nearest_even(%sum: i64) -> i16 {
  %mean = ondsp.round_div %sum {
    divisor = 3 : i64, pre_shift_left = 0 : i64,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (i64) -> i16
  return %mean : i16
}

// toward_zero IS the truncated quotient: no floor correction on the result.
// CHECK-LABEL: func.func @round_div_toward_zero
// CHECK: %[[TDIV:.*]] = arith.divsi
// CHECK-NOT: arith.select
// CHECK: arith.trunci
func.func @round_div_toward_zero(%sum: i32) -> i16 {
  %scaled = ondsp.round_div %sum {
    divisor = 7 : i64, pre_shift_left = 0 : i64,
    rounding = #ondsp.rounding<toward_zero>,
    overflow = #ondsp.overflow<wrap>
  } : (i32) -> i16
  return %scaled : i16
}

// The declared pre-scale widens into the exact carrier BEFORE shifting, so
// the shift cannot lose bits; the whole division then runs at that width.
// CHECK-LABEL: func.func @round_div_prescaled
// CHECK: arith.extsi %{{.*}} : i64 to i112
// CHECK: arith.shli {{.*}} : i112
// CHECK: arith.divsi {{.*}} : i112
// CHECK: arith.trunci {{.*}} : i112 to i32
func.func @round_div_prescaled(%value: i64) -> i32 {
  %result = ondsp.round_div %value {
    divisor = 999999937 : i64, pre_shift_left = 48 : i64,
    rounding = #ondsp.rounding<nearest_ties_positive>,
    overflow = #ondsp.overflow<saturate>
  } : (i64) -> i32
  return %result : i32
}

// Elementwise over a fixed vector, like every scalar-consumer policy op.
// CHECK-LABEL: func.func @round_div_vector
// CHECK: arith.divsi {{.*}} : vector<4xi32>
// CHECK: arith.remsi {{.*}} : vector<4xi32>
// CHECK: arith.trunci {{.*}} : vector<4xi32> to vector<4xi16>
func.func @round_div_vector(%values: vector<4xi32>) -> vector<4xi16> {
  %result = ondsp.round_div %values {
    divisor = 5 : i64, pre_shift_left = 0 : i64,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<saturate>
  } : (vector<4xi32>) -> vector<4xi16>
  return %result : vector<4xi16>
}

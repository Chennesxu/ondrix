// RUN: ondrix-opt %s --convert-ondsp-fixed-to-scalar | FileCheck %s
// RUN: ondrix-opt %s --convert-ondsp-fixed-to-scalar="sqrt-estimate" | FileCheck %s --check-prefix=ESTIMATE

// The scalar lowering is the exact unrolled bit-by-bit integer square root:
// sixteen candidate bits, each with square, compare, select; the
// nearest_even form adds one remainder-driven increment (no reachable tie),
// and the result clamps to 32767 before narrowing.

// The opt-in estimate mode replaces the sixteen candidate bits with one
// IEEE square root and four branchless correction steps; rounding and
// saturation are unchanged, so the result stays bit-identical.

// ESTIMATE-LABEL: func.func @sqrt_nearest
// ESTIMATE: math.sqrt {{.*}} : f64
// ESTIMATE-COUNT-6: arith.select
// ESTIMATE-NOT: arith.select

// CHECK-LABEL: func.func @sqrt_nearest
// CHECK-COUNT-18: arith.select
// CHECK-NOT: arith.select
// CHECK-NOT: ondsp.sqrt_fixed
// CHECK: arith.trunci {{.*}} : i64 to i16
func.func @sqrt_nearest(%input: i64) -> i16 {
  %root = ondsp.sqrt_fixed %input {
    rounding = #ondsp.rounding<nearest_even>
  } : (i64) -> i16
  return %root : i16
}

// ESTIMATE-LABEL: func.func @sqrt_floor
// ESTIMATE: math.sqrt
// ESTIMATE-COUNT-5: arith.select
// ESTIMATE-NOT: arith.select

// CHECK-LABEL: func.func @sqrt_floor
// CHECK-COUNT-17: arith.select
// CHECK-NOT: arith.select
// CHECK-NOT: ondsp.sqrt_fixed
func.func @sqrt_floor(%input: i64) -> i16 {
  %root = ondsp.sqrt_fixed %input {
    rounding = #ondsp.rounding<toward_negative>
  } : (i64) -> i16
  return %root : i16
}

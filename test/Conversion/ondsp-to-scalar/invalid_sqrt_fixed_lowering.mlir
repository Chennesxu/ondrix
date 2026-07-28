// RUN: not ondrix-opt %s --convert-ondsp-fixed-to-scalar --split-input-file 2>&1 | FileCheck %s

// Only the proven scalar i64 -> i16 profile lowers; other widths fail
// closed until a consumer defines their semantics and range proofs. A
// provably negative input violates the non-negative value domain and also
// fails closed instead of reaching the runtime clamp.

// CHECK: error: 'ondsp.sqrt_fixed' op fixed scalar lowering supports scalar i64 sqrt_fixed input

func.func @sqrt_narrow_input(%input: i32) -> i16 {
  %root = ondsp.sqrt_fixed %input {
    rounding = #ondsp.rounding<nearest_even>
  } : (i32) -> i16
  return %root : i16
}

// -----

// CHECK: error: 'ondsp.sqrt_fixed' op constant input is negative and outside the sqrt_fixed value domain

func.func @sqrt_negative_constant() -> i16 {
  %input = arith.constant -1 : i64
  %root = ondsp.sqrt_fixed %input {
    rounding = #ondsp.rounding<nearest_even>
  } : (i64) -> i16
  return %root : i16
}

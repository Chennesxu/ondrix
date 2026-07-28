// RUN: not ondrix-opt %s --convert-ondsp-fixed-to-scalar --split-input-file 2>&1 | FileCheck %s

// Only the proven scalar i64 -> i16 profile lowers; other widths fail
// closed until a consumer defines their semantics and range proofs.

// CHECK: error: 'ondsp.sqrt_fixed' op fixed scalar lowering supports scalar i64 sqrt_fixed input

func.func @sqrt_narrow_input(%input: i32) -> i16 {
  %root = ondsp.sqrt_fixed %input {
    rounding = #ondsp.rounding<nearest_even>
  } : (i32) -> i16
  return %root : i16
}

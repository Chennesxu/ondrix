// RUN: not ondrix-opt %s --convert-ondsp-to-ortumcore 2>&1 | FileCheck %s

func.func @output_type_escape(%lhs: i32, %rhs: i32)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  // CHECK: error: failed to legalize operation 'ortumcore.sat_add'
  %result = ortumcore.sat_add %lhs, %rhs : (i32, i32) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// RUN: not ondrix-opt %s --convert-ondsp-fixed-to-scalar 2>&1 | FileCheck %s

func.func @select_metadata(
    %condition: i1,
    %lhs: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    %rhs: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  // CHECK: 'arith.select' op attribute 'test.acc_type' contains a source accumulator type
  %selected = "arith.select"(%condition, %lhs, %rhs) {
    test.acc_type = [{nested = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>}]
  } : (i1, !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %selected : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

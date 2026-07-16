// RUN: not ondrix-opt %s --convert-ondsp-fixed-to-scalar 2>&1 | FileCheck %s

func.func @nested_select(
    %condition: i1,
    %lhs: tuple<!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>>,
    %rhs: tuple<!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>>)
    -> tuple<!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>> {
  // CHECK: nested accumulator containers are unsupported
  %selected = arith.select %condition, %lhs, %rhs : tuple<!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>>
  return %selected : tuple<!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>>
}

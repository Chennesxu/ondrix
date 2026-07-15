// RUN: not ondrix-opt %s --convert-ondsp-to-ortumcore 2>&1 | FileCheck %s

func.func @nested_accumulator_cast(%seed: i32) -> i32 {
  %tuple = builtin.unrealized_conversion_cast %seed : i32 to tuple<!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>>
  %result = builtin.unrealized_conversion_cast %tuple : tuple<!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>> to i32
  return %result : i32
}

// CHECK: failed to legalize operation 'builtin.unrealized_conversion_cast'

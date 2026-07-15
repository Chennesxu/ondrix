// RUN: not ondrix-opt %s -split-input-file --convert-ondsp-to-ortumcore 2>&1 | FileCheck %s

func.func @unsupported_accumulator_frac(
    %acc: !ondsp.acc<storage = i40, frac = 17, signed, update_overflow = saturate>)
    -> !ondsp.acc<storage = i40, frac = 17, signed, update_overflow = saturate> {
  // CHECK: unsupported accumulator type '!ondsp.acc<storage = i40, frac = 17, signed, update_overflow = saturate>'
  return %acc : !ondsp.acc<storage = i40, frac = 17, signed, update_overflow = saturate>
}

// -----

func.func @unsupported_accumulator_frac_below_q15_product(
    %acc: !ondsp.acc<storage = i40, frac = 29, signed, update_overflow = saturate>)
    -> !ondsp.acc<storage = i40, frac = 29, signed, update_overflow = saturate> {
  // CHECK: unsupported accumulator type '!ondsp.acc<storage = i40, frac = 29, signed, update_overflow = saturate>'
  return %acc : !ondsp.acc<storage = i40, frac = 29, signed, update_overflow = saturate>
}

// -----

func.func @unsupported_accumulator_frac_above_q15_product(
    %acc: !ondsp.acc<storage = i40, frac = 31, signed, update_overflow = saturate>)
    -> !ondsp.acc<storage = i40, frac = 31, signed, update_overflow = saturate> {
  // CHECK: unsupported accumulator type '!ondsp.acc<storage = i40, frac = 31, signed, update_overflow = saturate>'
  return %acc : !ondsp.acc<storage = i40, frac = 31, signed, update_overflow = saturate>
}

// -----

func.func @nested_unsupported_accumulator_frac(
    %acc: tuple<!ondsp.acc<storage = i40, frac = 29, signed, update_overflow = saturate>>) -> i32 {
  // CHECK: unsupported accumulator type '!ondsp.acc<storage = i40, frac = 29, signed, update_overflow = saturate>'
  %zero = arith.constant 0 : i32
  return %zero : i32
}

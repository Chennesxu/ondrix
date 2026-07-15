// RUN: not ondrix-opt %s -split-input-file --convert-ondsp-to-ortumcore 2>&1 | FileCheck %s

func.func @unsupported_accumulator_frac(
    %acc: !ondsp.acc<storage = i40, frac = 17, signed>)
    -> !ondsp.acc<storage = i40, frac = 17, signed> {
  // CHECK: unsupported accumulator type '!ondsp.acc<storage = i40, frac = 17, signed>'
  return %acc : !ondsp.acc<storage = i40, frac = 17, signed>
}

// -----

func.func @unsupported_accumulator_frac_below_q15_product(
    %acc: !ondsp.acc<storage = i40, frac = 29, signed>)
    -> !ondsp.acc<storage = i40, frac = 29, signed> {
  // CHECK: unsupported accumulator type '!ondsp.acc<storage = i40, frac = 29, signed>'
  return %acc : !ondsp.acc<storage = i40, frac = 29, signed>
}

// -----

func.func @unsupported_accumulator_frac_above_q15_product(
    %acc: !ondsp.acc<storage = i40, frac = 31, signed>)
    -> !ondsp.acc<storage = i40, frac = 31, signed> {
  // CHECK: unsupported accumulator type '!ondsp.acc<storage = i40, frac = 31, signed>'
  return %acc : !ondsp.acc<storage = i40, frac = 31, signed>
}

// -----

func.func @nested_unsupported_accumulator_frac(
    %acc: tuple<!ondsp.acc<storage = i40, frac = 29, signed>>) -> i32 {
  // CHECK: unsupported accumulator type '!ondsp.acc<storage = i40, frac = 29, signed>'
  %zero = arith.constant 0 : i32
  return %zero : i32
}

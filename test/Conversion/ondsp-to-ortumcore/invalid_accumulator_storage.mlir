// RUN: not ondrix-opt %s -split-input-file --convert-ondsp-to-ortumcore 2>&1 | FileCheck %s

func.func @unsupported_accumulator_signature(
    %acc: !ondsp.acc<storage = i64, frac = 30, signed>)
    -> !ondsp.acc<storage = i64, frac = 30, signed> {
  // CHECK: unsupported accumulator type '!ondsp.acc<storage = i64, frac = 30, signed>'
  return %acc : !ondsp.acc<storage = i64, frac = 30, signed>
}

// -----

func.func @unsupported_narrow_accumulator(
    %acc: !ondsp.acc<storage = i39, frac = 30, signed>)
    -> !ondsp.acc<storage = i39, frac = 30, signed> {
  // CHECK: unsupported accumulator type '!ondsp.acc<storage = i39, frac = 30, signed>'
  return %acc : !ondsp.acc<storage = i39, frac = 30, signed>
}

// -----

func.func @unsupported_unsigned_accumulator(
    %acc: !ondsp.acc<storage = i40, frac = 30, unsigned>)
    -> !ondsp.acc<storage = i40, frac = 30, unsigned> {
  // CHECK: unsupported accumulator type '!ondsp.acc<storage = i40, frac = 30, unsigned>'
  return %acc : !ondsp.acc<storage = i40, frac = 30, unsigned>
}

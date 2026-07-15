// RUN: not ondrix-opt %s -split-input-file --convert-ondsp-to-ortumcore 2>&1 | FileCheck %s

func.func @unsupported_accumulator_signature(
    %acc: !ondsp.acc<storage = i64, frac = 30, signed, update_overflow = saturate>)
    -> !ondsp.acc<storage = i64, frac = 30, signed, update_overflow = saturate> {
  // CHECK: unsupported accumulator type '!ondsp.acc<storage = i64, frac = 30, signed, update_overflow = saturate>'
  return %acc : !ondsp.acc<storage = i64, frac = 30, signed, update_overflow = saturate>
}

// -----

func.func @unsupported_narrow_accumulator(
    %acc: !ondsp.acc<storage = i39, frac = 30, signed, update_overflow = saturate>)
    -> !ondsp.acc<storage = i39, frac = 30, signed, update_overflow = saturate> {
  // CHECK: unsupported accumulator type '!ondsp.acc<storage = i39, frac = 30, signed, update_overflow = saturate>'
  return %acc : !ondsp.acc<storage = i39, frac = 30, signed, update_overflow = saturate>
}

// -----

func.func @unsupported_unsigned_accumulator(
    %acc: !ondsp.acc<storage = i40, frac = 30, unsigned, update_overflow = saturate>)
    -> !ondsp.acc<storage = i40, frac = 30, unsigned, update_overflow = saturate> {
  // CHECK: unsupported accumulator type '!ondsp.acc<storage = i40, frac = 30, unsigned, update_overflow = saturate>'
  return %acc : !ondsp.acc<storage = i40, frac = 30, unsigned, update_overflow = saturate>
}

// -----

func.func @unsupported_wrapping_accumulator(
    %acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap> {
  // CHECK: unsupported accumulator type '!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>'
  return %acc : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
}

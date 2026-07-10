// RUN: not ondrix-opt %s --convert-ondsp-to-ortumcore 2>&1 | FileCheck %s

func.func @unsupported_accumulator_signature(
    %acc: !ondsp.acc<storage = i64, frac = 30, signed>)
    -> !ondsp.acc<storage = i64, frac = 30, signed> {
  // CHECK: failed to legalize operation 'func.func'
  return %acc : !ondsp.acc<storage = i64, frac = 30, signed>
}

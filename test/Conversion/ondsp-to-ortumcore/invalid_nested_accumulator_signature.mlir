// RUN: not ondrix-opt %s --convert-ondsp-to-ortumcore 2>&1 | FileCheck %s

// CHECK: failed to legalize operation 'func.func'
func.func @nested_accumulator_signature(
    %arg: tuple<!ondsp.acc<storage = i40, frac = 30, signed>>) {
  return
}

// RUN: not ondrix-opt %s --allow-unregistered-dialect --convert-ondsp-to-ortumcore 2>&1 | FileCheck %s

func.func @unknown_nested_accumulator_result() {
  // CHECK: failed to legalize operation 'test.make_acc'
  %0 = "test.make_acc"() : () -> tuple<!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>>
  return
}

// RUN: not ondrix-opt %s -split-input-file --convert-ondsp-to-ortumcore 2>&1 | FileCheck %s

func.func @legacy_acc_init(%input: i16)
    -> !ondsp.acc<storage = i40, frac = 30, signed> {
  // CHECK: legacy accumulator import has no proven ortumcore equivalent
  %0 = ondsp.acc_init %input : (i16) -> !ondsp.acc<storage = i40, frac = 30, signed>
  return %0 : !ondsp.acc<storage = i40, frac = 30, signed>
}

// -----

func.func @legacy_acc_extract(
    %acc: !ondsp.acc<storage = i40, frac = 30, signed>) -> i32 {
  // CHECK: legacy accumulator extraction has no proven ortumcore equivalent
  %0 = ondsp.acc_extract %acc : (!ondsp.acc<storage = i40, frac = 30, signed>) -> i32
  return %0 : i32
}

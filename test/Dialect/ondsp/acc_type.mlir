// RUN: ondrix-opt %s | FileCheck %s

// CHECK-LABEL: func.func @acc_type
func.func @acc_type(%x: i16) -> i32 {
  // CHECK: !ondsp.acc<storage = i40, frac = 30, signed>
  %0 = ondsp.acc_init %x : (i16) -> !ondsp.acc<storage = i40, frac = 30, signed>
  %1 = ondsp.acc_extract %0 : (!ondsp.acc<storage = i40, frac = 30, signed>) -> i32
  return %1 : i32
}

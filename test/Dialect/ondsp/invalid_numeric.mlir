// RUN: not ondrix-opt %s 2>&1 | FileCheck %s

// CHECK: fixed numeric storage must be an integer type
func.func @invalid_fixed(%x: i16) -> i16 {
  %0 = ondsp.assume_numeric %x {numeric = #ondsp.fixed<signed, storage = f32, frac = 15>} : (i16) -> i16
  return %0 : i16
}

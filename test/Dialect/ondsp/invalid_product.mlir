// RUN: not ondrix-opt %s 2>&1 | FileCheck %s

// CHECK: error: invalid fixed-point product result selection
func.func @invalid_product(%x: i16) -> i16 {
  %0 = ondsp.assume_numeric %x {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<low>} : (i16) -> i16
  return %0 : i16
}

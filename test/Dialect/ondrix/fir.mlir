// RUN: ondrix-opt %s | FileCheck %s

// CHECK-LABEL: func.func @fir
func.func @fir(%x: memref<8xi16>, %h: memref<8xi16>) -> i32 {
  // CHECK: ondrix.fir
  %0 = ondrix.fir %x, %h {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (memref<8xi16>, memref<8xi16>) -> i32
  return %0 : i32
}

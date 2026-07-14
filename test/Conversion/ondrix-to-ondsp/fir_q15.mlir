// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s

func.func @fir_q15(%x: memref<8xi16>, %h: memref<8xi16>) -> i32 {
  %0 = ondrix.fir %x, %h {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (memref<8xi16>, memref<8xi16>) -> i32
  return %0 : i32
}

// CHECK-LABEL: func.func @fir_q15
// CHECK-NOT: ondrix.fir
// CHECK: ondsp.reduce_mac
// CHECK: #ondsp.fixed<signed, storage = i16, frac = 15>
// CHECK: #ondsp.product<full>

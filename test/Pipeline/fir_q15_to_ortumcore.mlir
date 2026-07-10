// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-to-ortumcore | FileCheck %s

func.func @fir_q15(%x: memref<16xi16>, %h: memref<8xi16>) -> i32 {
  %0 = ondrix.fir %x, %h {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (memref<16xi16>, memref<8xi16>) -> i32
  return %0 : i32
}

// CHECK-LABEL: func.func @fir_q15
// CHECK: ortumcore.acc_init
// CHECK: ortumcore.mac_add
// CHECK: ortumcore.acc_extract
// CHECK-NOT: ondsp.reduce_mac
// CHECK-NOT: ondrix.fir

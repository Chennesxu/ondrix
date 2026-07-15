// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s

func.func @fir_q15(%x: memref<8xi16>, %h: memref<8xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  %0 = ondrix.fir %x, %h {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (memref<8xi16>, memref<8xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %0 : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @fir_q15
// CHECK-NOT: ondrix.fir
// CHECK: %[[ZERO:.*]] = ondsp.acc_zero
// CHECK: ondsp.reduce_mac %[[ZERO]],
// CHECK: #ondsp.fixed<signed, storage = i16, frac = 15>
// CHECK: #ondsp.product<full>

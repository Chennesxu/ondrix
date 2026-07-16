// RUN: not ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-to-ortumcore 2>&1 | FileCheck %s

func.func @fir_q15(%x: memref<8xi16>, %h: memref<8xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
// CHECK: ordered fixed reduce_mac has no exact ortumcore lowering; scalarize to explicit accumulator updates or provide a proven target sequence
  %0 = ondrix.fir %x, %h {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (memref<8xi16>, memref<8xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %0 : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

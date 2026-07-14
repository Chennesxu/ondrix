// RUN: not ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-to-ortumcore 2>&1 | FileCheck %s

func.func @fir_q15(%x: memref<16xi16>, %h: memref<8xi16>) -> i32 {
  // CHECK: error: 'ondsp.reduce_mac' op shaped operands are not supported by ortumcore lowering; lower the reduction to an explicit loop of scalar MAC operations first
  %0 = ondrix.fir %x, %h {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (memref<16xi16>, memref<8xi16>) -> i32
  return %0 : i32
}

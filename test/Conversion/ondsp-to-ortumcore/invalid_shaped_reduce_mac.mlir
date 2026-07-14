// RUN: not ondrix-opt %s --convert-ondsp-to-ortumcore 2>&1 | FileCheck %s

func.func @shaped_reduce_mac(%lhs: memref<8xi16>, %rhs: memref<8xi16>) -> i32 {
  // CHECK: error: 'ondsp.reduce_mac' op shaped operands are not supported by ortumcore lowering; lower the reduction to an explicit loop of scalar MAC operations first
  %0 = ondsp.reduce_mac %lhs, %rhs {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (memref<8xi16>, memref<8xi16>) -> i32
  return %0 : i32
}

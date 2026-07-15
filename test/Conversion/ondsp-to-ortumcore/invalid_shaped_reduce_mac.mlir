// RUN: not ondrix-opt %s --convert-ondsp-to-ortumcore 2>&1 | FileCheck %s

func.func @shaped_reduce_mac(%lhs: memref<8xi16>, %rhs: memref<8xi16>) -> i32 {
  // CHECK: fixed reduce_mac has no ortumcore lowering until explicit accumulator update and export semantics are implemented
  %0 = ondsp.reduce_mac %lhs, %rhs {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (memref<8xi16>, memref<8xi16>) -> i32
  return %0 : i32
}

// RUN: not ondrix-opt %s --convert-ondsp-to-ortumcore 2>&1 | FileCheck %s

func.func @fixed_reduce_mac(%a: i16, %b: i16) -> i32 {
  // CHECK: fixed reduce_mac has no ortumcore lowering until explicit accumulator update and export semantics are implemented
  %0 = ondsp.reduce_mac %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (i16, i16) -> i32
  return %0 : i32
}

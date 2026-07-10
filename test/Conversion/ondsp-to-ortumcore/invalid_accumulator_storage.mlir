// RUN: not ondrix-opt %s --convert-ondsp-to-ortumcore 2>&1 | FileCheck %s

func.func @unsupported_accumulator_storage(
    %acc: !ondsp.acc<storage = i64, frac = 30, signed>, %a: i16, %b: i16)
    -> !ondsp.acc<storage = i64, frac = 30, signed> {
  // CHECK: ortumcore lowering requires a signed 40-bit ondsp accumulator
  %0 = ondsp.mac %acc, %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i64, frac = 30, signed>, i16, i16) -> !ondsp.acc<storage = i64, frac = 30, signed>
  return %0 : !ondsp.acc<storage = i64, frac = 30, signed>
}

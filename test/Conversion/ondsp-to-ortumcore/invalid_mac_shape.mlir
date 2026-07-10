// RUN: not ondrix-opt %s --convert-ondsp-to-ortumcore 2>&1 | FileCheck %s

func.func @bad_mac_operand_type(%acc: !ondsp.acc<storage = i40, frac = 30, signed>, %a: i32, %b: i32) -> !ondsp.acc<storage = i40, frac = 30, signed> {
  // CHECK: lhs type does not match fixed numeric storage type
  %0 = ondsp.mac %acc, %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed>, i32, i32) -> !ondsp.acc<storage = i40, frac = 30, signed>
  return %0 : !ondsp.acc<storage = i40, frac = 30, signed>
}

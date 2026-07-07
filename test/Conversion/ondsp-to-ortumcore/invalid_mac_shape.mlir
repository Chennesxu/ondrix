// RUN: not ondrix-opt %s --convert-ondsp-to-ortumcore 2>&1 | FileCheck %s

func.func @bad_mac_operand_type(%acc: !ortumcore.acc, %a: i32, %b: i32) -> !ortumcore.acc {
  // CHECK: operand type does not match fixed numeric storage type
  %0 = ondsp.mac %acc, %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>} : (!ortumcore.acc, i32, i32) -> !ortumcore.acc
  return %0 : !ortumcore.acc
}

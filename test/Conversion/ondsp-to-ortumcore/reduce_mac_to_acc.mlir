// RUN: ondrix-opt %s --convert-ondsp-to-ortumcore | FileCheck %s

func.func @reduce_mac(%a: i16, %b: i16) -> i32 {
  %0 = ondsp.reduce_mac %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (i16, i16) -> i32
  return %0 : i32
}

// CHECK-LABEL: func.func @reduce_mac
// CHECK: ortumcore.acc_init
// CHECK: ortumcore.mac_add
// CHECK: ortumcore.acc_extract
// CHECK-NOT: ondsp.reduce_mac

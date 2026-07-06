// RUN: ondrix-opt %s --convert-ondsp-to-ortumcore | FileCheck %s

func.func @q15_mac(%acc: !ondsp.acc<storage = i40, frac = 30, signed>,
                   %a: i16, %b: i16)
    -> !ondsp.acc<storage = i40, frac = 30, signed> {
  %0 = ondsp.mac %acc, %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>} : (!ondsp.acc<storage = i40, frac = 30, signed>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed>
  return %0 : !ondsp.acc<storage = i40, frac = 30, signed>
}

// CHECK-LABEL: func.func @q15_mac
// CHECK-NOT: ondsp.mac
// CHECK: ortumcore.dual_mac

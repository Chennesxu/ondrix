// RUN: not ondrix-opt %s --convert-ondsp-to-ortumcore 2>&1 | FileCheck %s

func.func @shaped_reduce_mac(%lhs: memref<8xi16>, %rhs: memref<8xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  %zero = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  // CHECK: ordered fixed reduce_mac has no exact ortumcore lowering; scalarize to explicit accumulator updates or provide a proven target sequence
  %0 = ondsp.reduce_mac %zero, %lhs, %rhs {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<8xi16>, memref<8xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %0 : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

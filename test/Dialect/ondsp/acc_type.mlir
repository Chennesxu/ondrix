// RUN: ondrix-opt %s | FileCheck %s

// CHECK-LABEL: func.func @acc_type
func.func @acc_type(%x: i16) -> i16 {
  // CHECK: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %0 = ondsp.acc_import %x {src = #ondsp.fixed<signed, storage = i16, frac = 15>} : (i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %1 = ondsp.acc_export %0 {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, rounding = #ondsp.rounding<nearest_even>, overflow = #ondsp.overflow<saturate>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %1 : i16
}

// CHECK-LABEL: func.func @wrap_acc_type
func.func @wrap_acc_type() -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap> {
  // CHECK: ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = wrap>
  %0 = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  return %0 : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
}

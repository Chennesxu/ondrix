// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar | FileCheck %s

func.func @fir_q15_to_scalar(
    %input: memref<8xi16>, %coeffs: memref<8xi16>) -> i16 {
  %acc = ondrix.fir %input, %coeffs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (memref<8xi16>, memref<8xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %result : i16
}

// CHECK-LABEL: func.func @fir_q15_to_scalar
// CHECK: scf.for
// CHECK: arith.muli
// CHECK-NOT: ondrix.
// CHECK-NOT: ondsp.

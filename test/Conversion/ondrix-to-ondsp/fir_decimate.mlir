// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s

// CHECK-LABEL: func.func @q15_decimate_by_two
// CHECK: %[[FACTOR:.*]] = arith.constant 2 : index
// CHECK: cf.assert {{.*}}, "fir_decimate requires at least one coefficient"
// CHECK: cf.assert {{.*}}, "fir_decimate input must cover one complete coefficient window"
// CHECK: arith.divui
// CHECK: cf.assert {{.*}}, "fir_decimate output length must equal floor((input length - coefficient length) / factor) plus one"
// CHECK: scf.for
// CHECK: %[[ORIGIN:.*]] = arith.muli {{.*}}, %[[FACTOR]]
// CHECK: scf.for
// CHECK: arith.addi %[[ORIGIN]]
// CHECK: ondsp.mac
// CHECK: ondsp.acc_export
// CHECK-NOT: ondrix.fir_decimate
func.func @q15_decimate_by_two(
    %input: tensor<?xi16>, %coeffs: tensor<?xi16>, %init: tensor<?xi16>)
    -> tensor<?xi16> {
  %result = ondrix.fir_decimate %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    factor = 2,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<?xi16>, tensor<?xi16>, tensor<?xi16>) -> tensor<?xi16>
  return %result : tensor<?xi16>
}

// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s

// CHECK-LABEL: func.func @q15_interpolate_by_two
// CHECK: %[[FACTOR:.*]] = arith.constant 2 : index
// CHECK: cf.assert {{.*}}, "fir_interpolate requires at least one input sample"
// CHECK: cf.assert {{.*}}, "fir_interpolate requires at least one coefficient"
// CHECK: arith.muli
// CHECK: arith.divui
// CHECK: cf.assert {{.*}}, "fir_interpolate result length multiplication must not overflow"
// CHECK: cf.assert {{.*}}, "fir_interpolate result length addition must not overflow"
// CHECK: cf.assert {{.*}}, "fir_interpolate output length must equal (input length - 1) * factor plus coefficient length"
// CHECK: scf.for %[[OUTPUT:[^ ]+]] =
// CHECK: scf.for %[[TAP:[^ ]+]] =
// CHECK: %[[COVERS:.*]] = arith.cmpi uge, %[[OUTPUT]], %[[TAP]]
// CHECK: %[[UPSAMPLED:.*]] = arith.subi %[[OUTPUT]], %[[TAP]]
// CHECK: %[[PHASE:.*]] = arith.remui %[[UPSAMPLED]], %[[FACTOR]]
// CHECK: %[[INPUT_INDEX:.*]] = arith.divui %[[UPSAMPLED]], %[[FACTOR]]
// CHECK: %[[CONTRIBUTES:.*]] = arith.andi %[[COVERS]],
// CHECK: scf.if %[[CONTRIBUTES]]
// CHECK: tensor.extract {{.*}}[%[[INPUT_INDEX]]]
// CHECK: tensor.extract {{.*}}[%[[TAP]]]
// CHECK: ondsp.mac
// CHECK: ondsp.acc_export
// CHECK-NOT: ondrix.fir_interpolate
func.func @q15_interpolate_by_two(
    %input: tensor<?xi16>, %coeffs: tensor<?xi16>, %init: tensor<?xi16>)
    -> tensor<?xi16> {
  %result = ondrix.fir_interpolate %input, %coeffs, %init {
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

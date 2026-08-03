// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s

// The f32 resampling profiles keep the fixed index relations and replace the
// accumulator machinery with the declared per-term events: no ondsp.mac and
// no acc_export survive on either path.

// CHECK-LABEL: func.func @f32_decimate_off
// CHECK: scf.for
// CHECK: %[[ORIGIN:.*]] = arith.muli
// CHECK: %[[SEED:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %[[SEED]]
// CHECK: arith.addi %[[ORIGIN]]
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK-NOT: ondsp.mac
// CHECK-NOT: ondsp.acc_export
func.func @f32_decimate_off(
    %input: tensor<?xf32>, %coeffs: tensor<?xf32>, %init: tensor<?xf32>)
    -> tensor<?xf32> {
  %result = ondrix.fir_decimate %input, %coeffs, %init {
    factor = 2,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %result : tensor<?xf32>
}

// CHECK-LABEL: func.func @f32_interpolate_fma
// The phase and bounds guard survives verbatim: skipping an inserted zero is
// a declared event, not a term the f32 contract lets a consumer materialize.
// CHECK: arith.remui
// CHECK: scf.if
// CHECK: math.fma
// CHECK-NOT: ondsp.mac
// CHECK-NOT: ondsp.acc_export
func.func @f32_interpolate_fma(
    %input: tensor<?xf32>, %coeffs: tensor<?xf32>, %init: tensor<?xf32>)
    -> tensor<?xf32> {
  %result = ondrix.fir_interpolate %input, %coeffs, %init {
    factor = 2,
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %result : tensor<?xf32>
}

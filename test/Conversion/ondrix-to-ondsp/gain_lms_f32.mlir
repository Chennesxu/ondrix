// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s

// The three gain declarations differ only in the permission the emitted
// multiply carries.
// CHECK-LABEL: func.func @f32_gain_off
// CHECK: arith.mulf %{{.*}}, %{{.*}} : f32
// CHECK-NOT: fastmath
func.func @f32_gain_off(%input: tensor<8xf32>) -> tensor<8xf32> {
  %result = ondrix.gain %input {
    fp_gain = 2.500000e-01 : f32,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %result : tensor<8xf32>
}

// CHECK-LABEL: func.func @f32_gain_fast
// CHECK: arith.mulf %{{[^ ]*}}, %{{[^ ]*}} : f32
func.func @f32_gain_fast(%input: tensor<8xf32>) -> tensor<8xf32> {
  %result = ondrix.gain %input {
    fp_gain = 2.500000e-01 : f32,
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %result : tensor<8xf32>
}

// Two sites are contract indexed: the tap reduction and each weight update.
// The error and the step scaling are single IEEE operations in every mode.
// The sample loop peels at min(K - 1, N) so the second region fetches taps
// without the prehistory guard.
// CHECK-LABEL: func.func @f32_lms_off
// CHECK-DAG: %[[N:.*]] = arith.constant 8 : index
// CHECK: %[[PRE:.*]]:2 = scf.for
// CHECK: arith.cmpi sge
// CHECK: arith.select
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: arith.subf
// CHECK: arith.mulf
// CHECK: scf.for
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK-NOT: ondsp.round_shift
// CHECK-NOT: ondsp.sat_cast
// CHECK: %[[MAIN:.*]]:2 = scf.for %{{.*}} to %[[N]] step %{{.*}} iter_args(%{{.*}} = %[[PRE]]#0, %{{.*}} = %[[PRE]]#1)
// CHECK-NOT: arith.cmpi
// CHECK-NOT: arith.maxsi
// CHECK-NOT: arith.select
// CHECK-NOT: ondsp.round_shift
// CHECK-NOT: ondsp.sat_cast
// CHECK: return %[[MAIN]]#1, %[[MAIN]]#0
func.func @f32_lms_off(%input: tensor<8xf32>, %desired: tensor<8xf32>, %weights: tensor<2xf32>)
    -> (tensor<8xf32>, tensor<2xf32>) {
  %error, %adapted = ondrix.lms %input, %desired, %weights {
    fp_step_size = 6.250000e-02 : f32,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<8xf32>, tensor<8xf32>, tensor<2xf32>) -> (tensor<8xf32>, tensor<2xf32>)
  return %error, %adapted : tensor<8xf32>, tensor<2xf32>
}

// CHECK-LABEL: func.func @f32_lms_fma
// CHECK-COUNT-2: math.fma
// CHECK-NOT: ondsp.round_shift
func.func @f32_lms_fma(%input: tensor<8xf32>, %desired: tensor<8xf32>, %weights: tensor<2xf32>)
    -> (tensor<8xf32>, tensor<2xf32>) {
  %error, %adapted = ondrix.lms %input, %desired, %weights {
    fp_step_size = 6.250000e-02 : f32,
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<8xf32>, tensor<8xf32>, tensor<2xf32>) -> (tensor<8xf32>, tensor<2xf32>)
  return %error, %adapted : tensor<8xf32>, tensor<2xf32>
}

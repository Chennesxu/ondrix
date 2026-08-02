// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" | FileCheck %s

// Only the reduction is contract indexed; the mean is one division by a
// representable constant and the root one correctly rounded square root.
// CHECK-LABEL: func.func @f32_rms
// CHECK: ondsp.reduce_mac
// CHECK-SAME: numeric = #ondsp.fp<format = f32, contract = off>
// CHECK: arith.divf
// CHECK: math.sqrt
func.func @f32_rms(%input: tensor<12xf32>) -> tensor<1xf32> {
  %result = ondrix.rms %input {
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<12xf32>) -> tensor<1xf32>
  return %result : tensor<1xf32>
}

// RUN: ondrix-opt %s --declare-ondrix-arguments-read-only | FileCheck %s
// RUN: ondrix-opt %s --pass-pipeline='builtin.module(convert-ondrix-to-ondsp,declare-ondrix-arguments-read-only,empty-tensor-to-alloc-tensor,one-shot-bufferize{bufferize-function-boundaries=true allow-return-allocs=true function-boundary-type-conversion=identity-layout-map create-deallocs=false})' | FileCheck %s --check-prefix=BUFFERIZED

// Every tensor argument is declared read-only; a scalar is not a buffer.
// CHECK-LABEL: func.func @f32_sos(
// CHECK-SAME: %arg0: tensor<8xf32> {bufferization.writable = false}
// CHECK-SAME: %arg1: tensor<2x5xf32> {bufferization.writable = false}
// CHECK-SAME: %arg2: tensor<2xf32> {bufferization.writable = false}
// CHECK-SAME: %arg3: tensor<2x2xf32> {bufferization.writable = false}

// The recursive filter updates its state in place. Read-only arguments make
// bufferization copy the state into the kernel's own buffer first, so the
// caller's input is never written.
// BUFFERIZED-LABEL: func.func @f32_sos(
// BUFFERIZED: %[[STATE:.*]] = memref.alloc() {{.*}} : memref<2x2xf32>
// BUFFERIZED: memref.copy %arg3, %[[STATE]]
// BUFFERIZED-NOT: memref.store {{.*}}, %arg3
func.func @f32_sos(%input: tensor<8xf32>, %coefficients: tensor<2x5xf32>, %scales: tensor<2xf32>,
                   %state: tensor<2x2xf32>) -> (tensor<8xf32>, tensor<2x2xf32>) {
  %output, %next = ondrix.sos_filter_tdf2 %input, %coefficients, %scales, %state {
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<8xf32>, tensor<2x5xf32>, tensor<2xf32>, tensor<2x2xf32>)
      -> (tensor<8xf32>, tensor<2x2xf32>)
  return %output, %next : tensor<8xf32>, tensor<2x2xf32>
}

// CHECK-LABEL: func.func @scalar_and_memref(
// CHECK-SAME: %arg0: i32,
// CHECK-SAME: %arg1: memref<4xi16>)
func.func @scalar_and_memref(%count: i32, %buffer: memref<4xi16>) {
  return
}

// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s
// RUN: ondrix-opt %s --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --canonicalize | FileCheck %s --check-prefix=BUFFER
// RUN: ondrix-opt %s --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --canonicalize --vectorize-ondsp-fixed-memref-reduce="vector-width=2" | FileCheck %s --check-prefix=VECTOR

// CHECK-LABEL: func.func @q15_convolution
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: tensor<?xi16>, %[[KERNEL:[a-zA-Z0-9_]+]]: tensor<?xi16>
// CHECK: %[[ONE:.*]] = arith.constant 1 : index
// CHECK: %[[KERNEL_LENGTH:.*]] = tensor.dim %[[KERNEL]]
// CHECK: scf.for
// CHECK: scf.for
// CHECK: %[[LAST:.*]] = arith.subi %[[KERNEL_LENGTH]], %[[ONE]]
// CHECK: %[[REVERSED:.*]] = arith.subi %[[LAST]],
// CHECK: tensor.extract {{.*}}[%[[REVERSED]]]
// CHECK: ondsp.mac
// CHECK: ondsp.acc_export
// CHECK-NOT: ondrix.conv1d
func.func @q15_convolution(
    %input: tensor<?xi16>, %kernel: tensor<?xi16>, %init: tensor<?xi16>)
    -> tensor<?xi16> {
  %result = ondrix.conv1d %input, %kernel, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    mode = #ondrix.conv1d_mode<convolution>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<?xi16>, tensor<?xi16>, tensor<?xi16>) -> tensor<?xi16>
  return %result : tensor<?xi16>
}

func.func @q15_correlation(
    %input: tensor<?xi16>, %kernel: tensor<?xi16>, %init: tensor<?xi16>)
    -> tensor<?xi16> {
  %result = ondrix.conv1d %input, %kernel, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    mode = #ondrix.conv1d_mode<correlation>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<?xi16>, tensor<?xi16>, tensor<?xi16>) -> tensor<?xi16>
  return %result : tensor<?xi16>
}

// CHECK-LABEL: func.func @f32_correlation
// CHECK: scf.for
// CHECK: scf.for
// CHECK-NOT: arith.subi
// CHECK: tensor.extract
// CHECK: tensor.extract
// CHECK: math.fma
// CHECK: tensor.insert
// CHECK-NOT: ondrix.conv1d
func.func @f32_correlation(
    %input: tensor<8xf32>, %kernel: tensor<3xf32>, %init: tensor<6xf32>)
    -> tensor<6xf32> {
  %result = ondrix.conv1d %input, %kernel, %init {
    mode = #ondrix.conv1d_mode<correlation>,
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<8xf32>, tensor<3xf32>, tensor<6xf32>) -> tensor<6xf32>
  return %result : tensor<6xf32>
}

// BUFFER-LABEL: func.func @q15_convolution
// BUFFER: %[[BUFFER_ONE:.*]] = arith.constant 1 : index
// BUFFER: %[[REVERSE_START:.*]] = arith.subi {{.*}}, %[[BUFFER_ONE]]
// BUFFER: %[[REVERSED:.*]] = memref.subview {{.*}}[%[[REVERSE_START]]] [{{.*}}] [-1]
// BUFFER: %[[WINDOW:.*]] = memref.subview {{.*}}[{{.*}}] [{{.*}}] [1]
// BUFFER: ondsp.reduce_mac {{.*}}, %[[WINDOW]], %[[REVERSED]]
// BUFFER-LABEL: func.func @q15_correlation
// BUFFER: %[[Q15_KERNEL_VIEW:.*]] = memref.subview {{.*}}[0] [{{.*}}] [1]
// BUFFER: ondsp.reduce_mac {{.*}}, {{.*}}, %[[Q15_KERNEL_VIEW]]
// BUFFER-LABEL: func.func @f32_correlation
// BUFFER: %[[KERNEL_VIEW:.*]] = memref.cast
// BUFFER: %[[CORR_WINDOW:.*]] = memref.subview
// BUFFER: ondsp.reduce_mac {{.*}}, {{.*}}, %[[KERNEL_VIEW]]

// VECTOR-LABEL: func.func @q15_convolution
// VECTOR-NOT: vector.load
// VECTOR: ondsp.reduce_mac
// VECTOR-LABEL: func.func @q15_correlation
// VECTOR: vector.load
// VECTOR: ondsp.reduce_mac {{.*}}vector<2xi16>

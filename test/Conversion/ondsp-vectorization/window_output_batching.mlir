// RUN: ondrix-opt %s --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --cse --canonicalize > %t.ordered.mlir
// RUN: ondrix-opt %t.ordered.mlir --vectorize-ondsp-fixed-decimate-outputs="vector-width=8" | FileCheck %s

// Unit-stride windows batch like decimation does, one output per lane; the
// stride-one span needs no lane extraction.

// Runtime coefficients: the lanes keep the declared saturating update.
// CHECK-LABEL: func.func @fir_runtime_coefficients
// CHECK: %[[BATCHED_END:.*]] = arith.constant 16 : index
// CHECK: scf.for %[[BLOCK:.*]] = %{{.*}} to %[[BATCHED_END]] step %{{.*}} {
// CHECK: ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>
// CHECK: vector.load %{{.*}}[%[[BLOCK]]] : memref<24xi16>, vector<8xi16>
// CHECK-NOT: vector.shuffle
// CHECK: ondsp.mac
// CHECK-COUNT-7: vector.load {{.*}} : memref<24xi16>, vector<8xi16>
// CHECK: vector.store {{.*}} : memref<17xi16>, vector<8xi16>
// CHECK: scf.for %{{.*}} = %[[BATCHED_END]] to
// CHECK: ondsp.reduce_mac
func.func @fir_runtime_coefficients(
    %input: tensor<24xi16>, %coeffs: tensor<8xi16>, %init: tensor<17xi16>) -> tensor<17xi16> {
  %result = ondrix.fir_filter %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<valid>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<24xi16>, tensor<8xi16>, tensor<17xi16>) -> tensor<17xi16>
  return %result : tensor<17xi16>
}

// A convolution flips its constant kernel: the taps read the reversed view
// one scalar at a time, the certified prefixes let the lanes wrap, and eight
// consecutive taps (sum of |c| under 2^16) accumulate in an i32 lane group.
// CHECK-LABEL: func.func @conv_constant_kernel_certified
// CHECK: scf.for
// CHECK: ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = wrap, lanes = 8>
// CHECK: ondsp.acc_zero : <storage = i32, frac = 30, signed, update_overflow = wrap, lanes = 8>
// CHECK: memref.load %{{.*}} : memref<16xi16, strided<[-1], offset: 15>>
// CHECK-COUNT-8: ondsp.mac
// CHECK: ondsp.acc_export {{.*}} -> vector<8xi32>
// CHECK: ondsp.acc_add_term {{.*}} lanes = 8>, vector<8xi32>)
// CHECK: ondsp.acc_zero : <storage = i32, frac = 30, signed, update_overflow = wrap, lanes = 8>
// CHECK-COUNT-8: ondsp.mac
// CHECK: ondsp.acc_add_term
// CHECK: ondsp.acc_export {{.*}} overflow = #ondsp.overflow<saturate>
// CHECK: vector.store {{.*}} : memref<32xi16>, vector<8xi16>
func.func @conv_constant_kernel_certified(%input: tensor<47xi16>) -> tensor<32xi16> {
  %kernel = arith.constant dense<[-10000, -7269, -4538, -1807, 924, 3655, 6386, 9117, -8153, -5422, -2691, 40, 2771, 5502, 8233, -9037]> : tensor<16xi16>
  %init = tensor.empty() : tensor<32xi16>
  %result = ondrix.conv1d %input, %kernel, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    mode = #ondrix.conv1d_mode<convolution>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<47xi16>, tensor<16xi16>, tensor<32xi16>) -> tensor<32xi16>
  return %result : tensor<32xi16>
}

// A wrapping accumulator needs no rail certificate; the same kernel still
// earns the i32 tap groups.
// CHECK-LABEL: func.func @conv_constant_kernel_wrapping
// CHECK: ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = wrap, lanes = 8>
// CHECK: ondsp.acc_zero : <storage = i32, frac = 30, signed, update_overflow = wrap, lanes = 8>
// CHECK-COUNT-8: ondsp.mac
// CHECK: ondsp.acc_export {{.*}} -> vector<8xi32>
// CHECK: ondsp.acc_add_term {{.*}} lanes = 8>, vector<8xi32>)
func.func @conv_constant_kernel_wrapping(%input: tensor<47xi16>) -> tensor<32xi16> {
  %kernel = arith.constant dense<[-10000, -7269, -4538, -1807, 924, 3655, 6386, 9117, -8153, -5422, -2691, 40, 2771, 5502, 8233, -9037]> : tensor<16xi16>
  %init = tensor.empty() : tensor<32xi16>
  %result = ondrix.conv1d %input, %kernel, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    mode = #ondrix.conv1d_mode<convolution>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<47xi16>, tensor<16xi16>, tensor<32xi16>) -> tensor<32xi16>
  return %result : tensor<32xi16>
}

// Eight full-scale taps sum to 2^33, which leaves an i33 accumulator, so no
// certificate exists and the lanes keep clamping.
// CHECK-LABEL: func.func @conv_constant_kernel_uncertified
// CHECK: ondsp.acc_zero : <storage = i33, frac = 30, signed, update_overflow = saturate, lanes = 8>
func.func @conv_constant_kernel_uncertified(%input: tensor<47xi16>) -> tensor<40xi16> {
  %kernel = arith.constant dense<[32767, -32768, 32767, -32768, 32767, -32768, 32767, -32768]> : tensor<8xi16>
  %init = tensor.empty() : tensor<40xi16>
  %result = ondrix.conv1d %input, %kernel, %init {
    accumulator = !ondsp.acc<storage = i33, frac = 30, signed, update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    mode = #ondrix.conv1d_mode<convolution>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<47xi16>, tensor<8xi16>, tensor<40xi16>) -> tensor<40xi16>
  return %result : tensor<40xi16>
}

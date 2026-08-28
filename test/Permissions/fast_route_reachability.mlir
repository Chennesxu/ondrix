// RUN: ondrix-opt %s --ondrix-default-pipeline="vector-bits=256 supports-f32-vector-fma=true" | FileCheck %s --implicit-check-not=ondsp.reduce_mac

// fp_fast_reduce_aot proves that a reduce_mac becomes a term-conserving
// horizontal schedule. It cannot prove that any given operation still forms
// the reduce_mac that reaches it, so a bufferization change that quietly
// stopped vectorizing an operation would leave that gate green. These pins
// close the composition: for each reduction shape a source operation presents,
// the fast route really does reach the rewrite and really does spend R.
//
// Grouped by the adapter shape each operation presents to the reduction, not
// by operation name, because that shape is what the rewrite accepts or
// refuses. Operations sharing a shape share the argument; a new operation
// needs a new pin only when it presents a new shape. Which permission each
// route then spends is gated separately, in the accounting tests.
//
// Read after full lowering: the permission record is a schedule-stage audit
// attribute that the LLVM conversion drops, and what has to hold here is that
// the composition survived to the object shape.
//
// Nothing is left half-lowered on any route. Stated as an implicit check on
// the RUN line: a CHECK-NOT placed before the first positive directive only
// guards the preamble, not each function after it.

// Adapter shape 1, whole contiguous memref: the operands are the reduction.
// CHECK-LABEL: llvm.func @shape_contiguous_dot
// CHECK: llvm.shufflevector %{{.*}} [0, 1, 2, 3] : vector<8xf32>
func.func @shape_contiguous_dot(%lhs: memref<32xf32>, %rhs: memref<32xf32>) -> f32 {
  %r = ondrix.dot %lhs, %rhs {
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (memref<32xf32>, memref<32xf32>) -> f32
  return %r : f32
}

// CHECK-LABEL: llvm.func @shape_contiguous_fir
// CHECK: llvm.shufflevector %{{.*}} [0, 1, 2, 3] : vector<8xf32>
func.func @shape_contiguous_fir(%window: memref<32xf32>, %coeffs: memref<32xf32>) -> f32 {
  %r = ondrix.fir %window, %coeffs {
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (memref<32xf32>, memref<32xf32>) -> f32
  return %r : f32
}

// Adapter shape 2, a sliding unit-stride window at a moving offset.
// CHECK-LABEL: llvm.func @shape_sliding_window_filter
// CHECK: llvm.shufflevector %{{.*}} [0, 1, 2, 3] : vector<8xf32>
func.func @shape_sliding_window_filter(%input: tensor<64xf32>, %coeffs: tensor<16xf32>)
    -> tensor<49xf32> {
  %init = tensor.empty() : tensor<49xf32>
  %r = ondrix.fir_filter %input, %coeffs, %init {
    boundary = #ondrix.fir_boundary<valid>,
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (tensor<64xf32>, tensor<16xf32>, tensor<49xf32>) -> tensor<49xf32>
  return %r : tensor<49xf32>
}

// The decimated window is the same shape at a stride-D offset, which is why
// it inherits the route rather than needing its own argument.
// CHECK-LABEL: llvm.func @shape_sliding_window_decimate
// CHECK: llvm.shufflevector %{{.*}} [0, 1, 2, 3] : vector<8xf32>
func.func @shape_sliding_window_decimate(%input: tensor<64xf32>, %coeffs: tensor<16xf32>)
    -> tensor<25xf32> {
  %init = tensor.empty() : tensor<25xf32>
  %r = ondrix.fir_decimate %input, %coeffs, %init {
    factor = 2 : i64,
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (tensor<64xf32>, tensor<16xf32>, tensor<25xf32>) -> tensor<25xf32>
  return %r : tensor<25xf32>
}

// CHECK-LABEL: llvm.func @shape_sliding_window_correlation
// CHECK: llvm.shufflevector %{{.*}} [0, 1, 2, 3] : vector<8xf32>
func.func @shape_sliding_window_correlation(%input: tensor<64xf32>, %kernel: tensor<16xf32>)
    -> tensor<49xf32> {
  %init = tensor.empty() : tensor<49xf32>
  %r = ondrix.conv1d %input, %kernel, %init {
    mode = #ondrix.conv1d_mode<correlation>,
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (tensor<64xf32>, tensor<16xf32>, tensor<49xf32>) -> tensor<49xf32>
  return %r : tensor<49xf32>
}

// Not an adapter shape: matmul presents its column axis, which the
// order-preserving batching takes first under the declared vector FMA, so no
// reduction reaches the horizontal route. Pinned here because that is exactly
// the composition a bufferization change could break silently.
// CHECK-LABEL: llvm.func @shape_matrix_column_tile
// CHECK: llvm.intr.fma{{.*}} -> vector<8xf32>
// CHECK-NOT: llvm.shufflevector %{{.*}} [0, 1, 2, 3] : vector<8xf32>
func.func @shape_matrix_column_tile(%a: tensor<16x16xf32>, %b: tensor<16x16xf32>)
    -> tensor<16x16xf32> {
  %r = ondrix.matmul %a, %b {
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
  return %r : tensor<16x16xf32>
}

// Adapter shape 3, one operand used twice.
// CHECK-LABEL: llvm.func @shape_self_product
// CHECK: llvm.shufflevector %{{.*}} [0, 1, 2, 3] : vector<8xf32>
func.func @shape_self_product(%input: tensor<32xf32>) -> tensor<1xf32> {
  %r = ondrix.rms %input {
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (tensor<32xf32>) -> tensor<1xf32>
  return %r : tensor<1xf32>
}

// Adapter shape 4, a reversed subview. Refused, so the route spends F on the
// scalar fused chain instead — the one shape whose refusal is the point.
// CHECK-LABEL: llvm.func @shape_reversed_subview
// CHECK-NOT: llvm.shufflevector %{{.*}} [0, 1, 2, 3] : vector<8xf32>
func.func @shape_reversed_subview(%input: tensor<64xf32>, %kernel: tensor<16xf32>)
    -> tensor<49xf32> {
  %init = tensor.empty() : tensor<49xf32>
  %r = ondrix.conv1d %input, %kernel, %init {
    mode = #ondrix.conv1d_mode<convolution>,
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (tensor<64xf32>, tensor<16xf32>, tensor<49xf32>) -> tensor<49xf32>
  return %r : tensor<49xf32>
}

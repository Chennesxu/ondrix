// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s --check-prefix=SCALAR --implicit-check-not=fastmath

// Which permission a schedule spends leaves no fast-math flag behind, so it is
// recorded instead. Getting the enum wrong, or dropping the record, changes
// this output; the attribute is discardable and does not reach the object.
// The routes that spend R are in the companion horizontal test.
//
// These are also the structural membership pins for the four routes that
// select a fused chain: each builds explicit fused events in declared order,
// none carries a fast-math flag, and the record says which permission bought
// them. The absence of fast-math flags is an implicit check on the RUN line,
// because a SCALAR-NOT before the first label would only guard the preamble.

// A reduction on the tensor path never reaches the horizontal rewrite, so R
// goes unused and the fused chain spends F.
// SCALAR-LABEL: func.func @tensor_route
// SCALAR: math.fma {{.*}}used_permissions = ["fuse_multiply_add"]
func.func @tensor_route(%input: tensor<8xf32>) -> tensor<8xf32> {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fp<format = f32, contract = fast>,
    output_numeric = #ondsp.fp<format = f32, contract = fast>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %result : tensor<8xf32>
}

// A lone product has nothing to spend on any target.
// SCALAR-LABEL: func.func @elementwise_route
// SCALAR-NOT: ondsp.fast_selection
// SCALAR: arith.mulf
func.func @elementwise_route(%input: tensor<8xf32>) -> tensor<8xf32> {
  %result = ondrix.gain %input {
    fp_gain = 2.500000e-01 : f32,
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %result : tensor<8xf32>
}

// The other three fused-chain routes. lms has two contract-indexed sites, the
// tap reduction and the weight update.
// SCALAR-LABEL: func.func @lms_route
// SCALAR-COUNT-2: math.fma {{.*}}used_permissions = ["fuse_multiply_add"]
func.func @lms_route(%input: tensor<8xf32>, %desired: tensor<8xf32>, %weights: tensor<2xf32>)
    -> (tensor<8xf32>, tensor<2xf32>) {
  %error, %adapted = ondrix.lms %input, %desired, %weights {
    fp_step_size = 6.250000e-02 : f32,
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (tensor<8xf32>, tensor<8xf32>, tensor<2xf32>) -> (tensor<8xf32>, tensor<2xf32>)
  return %error, %adapted : tensor<8xf32>, tensor<2xf32>
}

// SCALAR-LABEL: func.func @interpolate_route
// SCALAR: math.fma {{.*}}used_permissions = ["fuse_multiply_add"]
func.func @interpolate_route(%input: tensor<4xf32>, %coeffs: tensor<3xf32>) -> tensor<9xf32> {
  %init = tensor.empty() : tensor<9xf32>
  %r = ondrix.fir_interpolate %input, %coeffs, %init {
    factor = 2 : i64,
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (tensor<4xf32>, tensor<3xf32>, tensor<9xf32>) -> tensor<9xf32>
  return %r : tensor<9xf32>
}

// The reversed subview is refused by the batching rewrite, so this route also
// lands on the fused chain; its refusal is pinned in fast_route_reachability.
// SCALAR-LABEL: func.func @reversed_subview_route
// SCALAR: math.fma {{.*}}used_permissions = ["fuse_multiply_add"]
func.func @reversed_subview_route(%input: tensor<12xf32>, %kernel: tensor<4xf32>)
    -> tensor<9xf32> {
  %init = tensor.empty() : tensor<9xf32>
  %r = ondrix.conv1d %input, %kernel, %init {
    mode = #ondrix.conv1d_mode<convolution>,
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (tensor<12xf32>, tensor<4xf32>, tensor<9xf32>) -> tensor<9xf32>
  return %r : tensor<9xf32>
}

// Neither exact contract spends anything, in any route.
// SCALAR-LABEL: func.func @exact_routes
// SCALAR-NOT: ondsp.fast_selection
func.func @exact_routes(%input: tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
  %off = ondrix.dct %input {
    input_numeric = #ondsp.fp<format = f32, contract = off>,
    output_numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<8xf32>) -> tensor<8xf32>
  %fma = ondrix.dct %input {
    input_numeric = #ondsp.fp<format = f32, contract = fma>,
    output_numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %off, %fma : tensor<8xf32>, tensor<8xf32>
}

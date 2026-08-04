// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s --check-prefix=SCALAR

// Which permission a schedule spends leaves no fast-math flag behind, so it is
// recorded instead. Getting the enum wrong, or dropping the record, changes
// this output; the attribute is discardable and does not reach the object.
// The routes that spend R are in the companion horizontal test.

// A reduction on the tensor path never reaches the horizontal rewrite, so R
// goes unused and the fused chain spends F.
// SCALAR-LABEL: func.func @tensor_route
// SCALAR: math.fma {{.*}}{ondsp.fast_used = "fuse_multiply_add"}
func.func @tensor_route(%input: tensor<8xf32>) -> tensor<8xf32> {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fp<format = f32, contract = fast>,
    output_numeric = #ondsp.fp<format = f32, contract = fast>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %result : tensor<8xf32>
}

// A lone product has nothing to spend on any target.
// SCALAR-LABEL: func.func @elementwise_route
// SCALAR-NOT: ondsp.fast_used
// SCALAR: arith.mulf
func.func @elementwise_route(%input: tensor<8xf32>) -> tensor<8xf32> {
  %result = ondrix.gain %input {
    fp_gain = 2.500000e-01 : f32,
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %result : tensor<8xf32>
}

// Neither exact contract spends anything, in any route.
// SCALAR-LABEL: func.func @exact_routes
// SCALAR-NOT: ondsp.fast_used
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

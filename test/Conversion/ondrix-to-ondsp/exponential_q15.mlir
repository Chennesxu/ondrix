// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --split-input-file | FileCheck %s

// The tables are compile-time constants under the same tie guard the design
// tables use, and each has 129 entries so the interpolation's upper
// neighbour is the exact binade endpoint rather than a wrapped index.

// CHECK-LABEL: func.func @log2
// CHECK: arith.constant dense<{{.*}}> : tensor<129xi32>
// CHECK: math.ctlz
// CHECK: ondsp.round_shift {{.*}}post_shift_right = 8, rounding = nearest_even
// The pole is a declared value, not a saturation the caller cannot see.
// CHECK: arith.select
func.func @log2(%a: tensor<8xi16>) -> tensor<8xi16> {
  %0 = ondrix.log2 %a {
    numeric = #ondsp.fixed<unsigned, storage = i16, frac = 16>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 11>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>) -> tensor<8xi16>
  return %0 : tensor<8xi16>
}

// -----

// Two boundaries: the mantissa interpolation is a round_shift, the binade
// placement is written out because its amount depends on the input.
// CHECK-LABEL: func.func @exp2
// CHECK: arith.constant dense<{{.*}}> : tensor<129xi32>
// CHECK: ondsp.round_shift {{.*}}post_shift_right = 4, rounding = nearest_even
// CHECK: arith.shrsi
// CHECK-NOT: ondsp.round_shift
func.func @exp2(%a: tensor<8xi16>) -> tensor<8xi16> {
  %0 = ondrix.exp2 %a {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 11>,
    output_numeric = #ondsp.fixed<unsigned, storage = i16, frac = 16>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>) -> tensor<8xi16>
  return %0 : tensor<8xi16>
}

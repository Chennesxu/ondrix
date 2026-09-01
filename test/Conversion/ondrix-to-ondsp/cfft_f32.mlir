// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s --implicit-check-not=ondsp.
// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s --check-prefix=EXACT

// Emitting no ondsp operation is the claim: an exact-format transform has no
// requantization boundary. Constants are pinned as the f32 DCT table is.

// CHECK-LABEL: func.func @f32_cfft16
// CHECK-DAG: arith.constant 0.923879504 : f32
// CHECK-DAG: arith.constant -0.923879504 : f32
// CHECK-DAG: arith.constant 0.707106769 : f32
// CHECK-DAG: arith.constant -0.707106769 : f32
// CHECK-DAG: arith.constant 0.382683426 : f32
// CHECK-DAG: arith.constant -0.382683426 : f32

// Forty products where multiplying every leg would emit 128: the twenty-two
// legs whose twiddle is exactly one or exactly -j carry no product at all.
// EXACT-LABEL: func.func @f32_cfft16
// EXACT-COUNT-40: arith.mulf
// EXACT-NOT: arith.mulf
// EXACT-NOT: math.fma
func.func @f32_cfft16(%input: tensor<32xf32>) -> tensor<32xf32> {
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<forward>,
    layout = #ondsp.cx_layout<interleaved>,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<32xf32>) -> tensor<32xf32>
  return %result : tensor<32xf32>
}

// The single 1/16 is the whole inverse normalization; twenty fused events are
// the ten complex products of the non-exact legs.
// CHECK-LABEL: func.func @f32_icfft16
// CHECK-DAG: arith.constant 6.250000e-02 : f32
// EXACT-LABEL: func.func @f32_icfft16
// EXACT-COUNT-20: math.fma
// EXACT-NOT: math.fma
func.func @f32_icfft16(%input: tensor<32xf32>) -> tensor<32xf32> {
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<inverse>,
    layout = #ondsp.cx_layout<interleaved>,
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<32xf32>) -> tensor<32xf32>
  return %result : tensor<32xf32>
}

// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s

// A three-deep loop nest: exact i64 K-accumulation innermost, then one
// nearest-even saturating ondsp.round_shift per output element — the
// FIR dot-product contract shape extended to a rank-2 result.

// CHECK-LABEL: func.func @matmul_q15
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: arith.muli
// CHECK: ondsp.round_shift
// CHECK-NOT: ondsp.round_shift
// CHECK: tensor.insert
// CHECK-NOT: ondrix.matmul
func.func @matmul_q15(%a: tensor<4x8xi16>, %b: tensor<8x3xi16>) -> tensor<4x3xi16> {
  %c = ondrix.matmul %a, %b {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4x8xi16>, tensor<8x3xi16>) -> tensor<4x3xi16>
  return %c : tensor<4x3xi16>
}

// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s --check-prefix=F32-OFF
// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s --check-prefix=F32-FMA

// The f32 tensor path runs the same loop nest over the declared per-term
// events and stores the reduction result directly: no boundary follows it.
// F32-OFF-LABEL: func.func @f32_matmul_off
// F32-OFF: arith.mulf
// F32-OFF: arith.addf
// F32-OFF-NOT: ondsp.round_shift
func.func @f32_matmul_off(%a: tensor<2x3xf32>, %b: tensor<3x2xf32>) -> tensor<2x2xf32> {
  %c = ondrix.matmul %a, %b {
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
  return %c : tensor<2x2xf32>
}

// F32-FMA-LABEL: func.func @f32_matmul_fma
// F32-FMA: math.fma
// F32-FMA-NOT: ondsp.round_shift
func.func @f32_matmul_fma(%a: tensor<2x3xf32>, %b: tensor<3x2xf32>) -> tensor<2x2xf32> {
  %c = ondrix.matmul %a, %b {
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
  return %c : tensor<2x2xf32>
}

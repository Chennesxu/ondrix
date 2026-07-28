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

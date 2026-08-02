// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s

// One exact square-accumulation loop, then exactly the two declared
// boundaries: the nearest-even mean by log2(N) into i32 (unreachable
// saturation) and the integer square root of the structurally
// nonnegative mean.

// The op's rounding attribute governs ONLY the root; the mean boundary
// is always nearest_even. Both routings are pinned so a dropped or
// misrouted attribute cannot pass silently.

// CHECK-LABEL: func.func @rms16_q15
// CHECK: scf.for
// CHECK: arith.muli
// CHECK: ondsp.round_shift
// CHECK-SAME: post_shift_right = 4
// CHECK-SAME: rounding = nearest_even
// CHECK-SAME: saturate_to = i32
// CHECK: ondsp.sqrt_fixed
// CHECK-SAME: rounding = #ondsp.rounding<nearest_even>
// CHECK-NOT: ondrix.rms
func.func @rms16_q15(%input: tensor<16xi16>) -> tensor<1xi16> {
  %result = ondrix.rms %input {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<16xi16>) -> tensor<1xi16>
  return %result : tensor<1xi16>
}

// CHECK-LABEL: func.func @rms16_floor
// CHECK: ondsp.round_shift
// CHECK-SAME: rounding = nearest_even
// CHECK: ondsp.sqrt_fixed
// CHECK-SAME: rounding = #ondsp.rounding<toward_negative>
func.func @rms16_floor(%input: tensor<16xi16>) -> tensor<1xi16> {
  %result = ondrix.rms %input {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>
  } : (tensor<16xi16>) -> tensor<1xi16>
  return %result : tensor<1xi16>
}

// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s --check-prefix=F32

// Only the reduction is contract indexed; the mean and the root are single
// operations and carry no fast-math flag under any contract.
// F32-LABEL: func.func @f32_rms
// F32: math.fma
// F32: arith.divf
// F32-NOT: fastmath
// F32: math.sqrt
// F32-NOT: ondsp.sqrt_fixed
func.func @f32_rms(%input: tensor<10xf32>) -> tensor<1xf32> {
  %result = ondrix.rms %input {
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<10xf32>) -> tensor<1xf32>
  return %result : tensor<1xf32>
}

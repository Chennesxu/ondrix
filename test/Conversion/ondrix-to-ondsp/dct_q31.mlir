// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s
// RUN: ondrix-opt %s --convert-ondrix-to-ondsp="preserve-bufferizable-reductions=true" | FileCheck %s --check-prefix=KEEP

// Q31 now has a bufferized reduce_mac spelling, so the contract form survives
// for it exactly as it does for Q15; expanding here would strand the schedule
// stage with no reduction to claim.
// KEEP-LABEL: func.func @dct8_q31
// KEEP: ondrix.dct
// KEEP-NOT: ondsp.round_shift

// The Q31 profile narrows each product before it joins the row sum. Two
// extents derive two different product shifts, so a lowering that pinned one
// fails the other; the export shift is the SAME 32 at both, because the
// narrowing already absorbed the extent dependence that leaves Q15 at 16 + m.

// CHECK-LABEL: func.func @dct8_q31
// CHECK: arith.muli
// CHECK: ondsp.round_shift {{.*}}post_shift_right = 3, rounding = toward_negative, overflow = saturate, saturate_to = i64
// CHECK: ondsp.round_shift {{.*}}post_shift_right = 32, rounding = nearest_even, overflow = saturate, saturate_to = i32
// CHECK-NOT: ondrix.dct
func.func @dct8_q31(%input: tensor<8xi32>) -> tensor<8xi32> {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    output_numeric = #ondsp.fixed<signed, storage = i32, frac = 27>,
    product_rounding = #ondsp.rounding<toward_negative>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi32>) -> tensor<8xi32>
  return %result : tensor<8xi32>
}

// CHECK-LABEL: func.func @dct64_q31
// CHECK: ondsp.round_shift {{.*}}post_shift_right = 6, {{.*}}saturate_to = i64
// CHECK: ondsp.round_shift {{.*}}post_shift_right = 32, {{.*}}saturate_to = i32
func.func @dct64_q31(%input: tensor<64xi32>) -> tensor<64xi32> {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    output_numeric = #ondsp.fixed<signed, storage = i32, frac = 24>,
    product_rounding = #ondsp.rounding<nearest_even>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<64xi32>) -> tensor<64xi32>
  return %result : tensor<64xi32>
}

// The Q15 profile is untouched: no product boundary at any admitted extent,
// and the export still carries the extent through 16 + m.
// CHECK-LABEL: func.func @dct8_q15
// CHECK-NOT: saturate_to = i64
// CHECK: ondsp.round_shift {{.*}}post_shift_right = 19, {{.*}}saturate_to = i16
func.func @dct8_q15(%input: tensor<8xi16>) -> tensor<8xi16> {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 11>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>) -> tensor<8xi16>
  return %result : tensor<8xi16>
}

// The raw-high profile floors each product by the storage width (a
// toward-negative shift by 32 in the i64 carrier) and the export shifts the
// frac-30 row sum by m = 3 onto the same frac-27 reading.
// CHECK-LABEL: func.func @dct8_q31_raw_high
// CHECK: ondsp.round_shift {{.*}}post_shift_right = 32, rounding = toward_negative, overflow = saturate, saturate_to = i64
// CHECK: ondsp.round_shift {{.*}}post_shift_right = 3, rounding = nearest_even, overflow = saturate, saturate_to = i32
// CHECK-NOT: ondrix.dct
func.func @dct8_q31_raw_high(%input: tensor<8xi32>) -> tensor<8xi32> {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    output_numeric = #ondsp.fixed<signed, storage = i32, frac = 27>,
    product = #ondsp.product<high_raw>
  } : (tensor<8xi32>) -> tensor<8xi32>
  return %result : tensor<8xi32>
}

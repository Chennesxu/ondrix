// RUN: ondrix-opt %s --convert-ondsp-fixed-to-scalar | FileCheck %s --implicit-check-not=ondsp.
// RUN: ondrix-opt %s --convert-ondsp-fixed-to-scalar="specialize-canonical-twiddles" | FileCheck %s --check-prefix=GENERAL --implicit-check-not=ondsp.

// The Q31 cross products br*wr - bi*wi reach +-2^63 and would wrap an i64
// carrier, so the exact product carrier is i128 and the sum carrier is i33.
// The i128 operations must be visible here: a silent narrowing to i64 would
// still produce plausible output for small inputs.
func.func @cx_butterfly_q31(%a: i64, %b: i64, %tw: i64) -> (i64, i64) {
  %o0, %o1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 31, rounding = nearest_even, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i32>
  } : (i64, i64, i64) -> (i64, i64)
  return %o0, %o1 : i64, i64
}

// CHECK-LABEL: func.func @cx_butterfly_q31
// Unpack: real is the low i32, imaginary the logical high i32.
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.shrui %{{.*}} : i64
// Exact i128 complex product.
// CHECK: arith.extsi %{{.*}} : i32 to i128
// CHECK: arith.muli %{{.*}} : i128
// CHECK: arith.muli %{{.*}} : i128
// CHECK: arith.subi %{{.*}} : i128
// CHECK: arith.addi %{{.*}} : i128
// The product requantization: nearest-even shift 31 out of the i128 carrier,
// saturating into i32.
// CHECK: arith.shrsi %{{.*}} : i128
// CHECK: arith.trunci %{{.*}} : i128 to i32
// Exact i33 sum carrier for a +- b*w, then the shift-1 output scale.
// CHECK: arith.extsi %{{.*}} : i32 to i33
// CHECK: arith.addi %{{.*}} : i33
// CHECK: arith.subi %{{.*}} : i33
// CHECK: arith.trunci %{{.*}} : i33 to i32
// Repack into the i64 container.
// CHECK: arith.shli %{{.*}} : i64
// CHECK: arith.ori %{{.*}} : i64

// Canonical-twiddle specialization is proven for packed Q15 only, so the same
// opt-in flag must leave the Q31 general product path in place.
// GENERAL-LABEL: func.func @cx_butterfly_q31
// GENERAL: arith.muli %{{.*}} : i128
// GENERAL: arith.muli %{{.*}} : i128
// GENERAL: arith.muli %{{.*}} : i128
// GENERAL: arith.muli %{{.*}} : i128

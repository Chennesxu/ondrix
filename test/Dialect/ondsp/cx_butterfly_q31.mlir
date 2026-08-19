// RUN: ondrix-opt %s | FileCheck %s

func.func @cx_butterfly_q31(%a: i64, %b: i64, %tw: i64) -> (i64, i64) {
  // CHECK: ondsp.cx_butterfly
  // CHECK-SAME: layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>
  // CHECK-SAME: numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
  // CHECK-SAME: output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i32>
  // CHECK-SAME: product = #ondsp.product<full>
  // CHECK-SAME: product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 31, rounding = nearest_even, overflow = saturate, saturate_to = i32>
  %o0, %o1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 31, rounding = nearest_even, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i32>
  } : (i64, i64, i64) -> (i64, i64)
  return %o0, %o1 : i64, i64
}

// The raw-high term is the Q31 scalar target's own product selection: every
// cross term narrows before the combine, and the product scale shifts LEFT by
// one because the raw high half lands a bit short of the component position.
func.func @cx_butterfly_q31_raw_high(%a: i64, %b: i64, %tw: i64) -> (i64, i64) {
  // CHECK: ondsp.cx_butterfly
  // CHECK-SAME: product = #ondsp.product<high_raw>
  // CHECK-SAME: product_scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 0, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  %o0, %o1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>,
    product_scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 0, rounding = toward_negative, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  } : (i64, i64, i64) -> (i64, i64)
  return %o0, %o1 : i64, i64
}

// The Q31 profile admits the same fixed Vector lane batching as Q15; only the
// container width differs.
func.func @cx_butterfly_q31_vector(
    %a: vector<2xi64>, %b: vector<2xi64>, %tw: vector<2xi64>)
    -> (vector<2xi64>, vector<2xi64>) {
  // CHECK: ondsp.cx_butterfly
  // CHECK-SAME: (vector<2xi64>, vector<2xi64>, vector<2xi64>) -> (vector<2xi64>, vector<2xi64>)
  %o0, %o1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 31, rounding = nearest_even, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i32>
  } : (vector<2xi64>, vector<2xi64>, vector<2xi64>) -> (vector<2xi64>, vector<2xi64>)
  return %o0, %o1 : vector<2xi64>, vector<2xi64>
}

// The raw-high selection batches over the same Vector carrier: the policy
// check is per element, so the lane shape adds nothing to admit.
func.func @cx_butterfly_q31_raw_high_vector(
    %a: vector<2xi64>, %b: vector<2xi64>, %tw: vector<2xi64>)
    -> (vector<2xi64>, vector<2xi64>) {
  // CHECK: ondsp.cx_butterfly
  // CHECK-SAME: product = #ondsp.product<high_raw>
  // CHECK-SAME: (vector<2xi64>, vector<2xi64>, vector<2xi64>) -> (vector<2xi64>, vector<2xi64>)
  %o0, %o1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>,
    product_scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 0, rounding = toward_negative, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  } : (vector<2xi64>, vector<2xi64>, vector<2xi64>) -> (vector<2xi64>, vector<2xi64>)
  return %o0, %o1 : vector<2xi64>, vector<2xi64>
}

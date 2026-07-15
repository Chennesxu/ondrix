// RUN: not ondrix-opt %s -split-input-file --convert-ondsp-to-ortumcore 2>&1 | FileCheck %s

func.func @vector_butterfly_requires_vector_lowering(
    %a: vector<2xi32>, %b: vector<2xi32>, %tw: vector<2xi32>)
    -> (vector<2xi32>, vector<2xi32>) {
  // CHECK: ortumcore butterfly lowering requires scalar packed i32 operands
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>, numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>, scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = toward_negative, overflow = saturate, saturate_to = i16>} : (vector<2xi32>, vector<2xi32>, vector<2xi32>) -> (vector<2xi32>, vector<2xi32>)
  return %0, %1 : vector<2xi32>, vector<2xi32>
}

// -----

func.func @vector_butterfly_results_require_vector_lowering(
    %a: i32, %b: i32, %tw: i32) -> (vector<2xi32>, vector<2xi32>) {
  // CHECK: ortumcore butterfly lowering requires scalar packed i32 results
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>, numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>, scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = toward_negative, overflow = saturate, saturate_to = i16>} : (i32, i32, i32) -> (vector<2xi32>, vector<2xi32>)
  return %0, %1 : vector<2xi32>, vector<2xi32>
}

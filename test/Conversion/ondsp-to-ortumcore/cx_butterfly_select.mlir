// RUN: ondrix-opt %s --convert-ondsp-cx-butterfly-to-ortumcore | FileCheck %s

// A target-inventory butterfly becomes the conjugated product plus the plain
// packed butterfly; the twiddle constant is conjugated exactly.
func.func @selects_floor_wrap(%a: i32, %b: i32) -> (i32, i32) {
  %tw = arith.constant 196610 : i32  // imag 3, real 2
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = toward_negative, overflow = wrap, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = wrap, saturate_to = i16>
  } : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

// CHECK-LABEL: func.func @selects_floor_wrap(
// CHECK-SAME: %[[A:.*]]: i32, %[[B:.*]]: i32)
// CHECK: %[[CONJ:.*]] = arith.constant -196606 : i32
// CHECK: %[[T:.*]] = ortumcore.cx_mul_conj %[[B]], %[[CONJ]]
// CHECK-SAME: layout = #ortumcore<cx_layout imag_hi>
// CHECK-SAME: overflow = #ortumcore<cx_overflow wrap>
// CHECK-SAME: rounding = #ortumcore<cx_rounding toward_negative>
// CHECK-SAME: shift = 15
// CHECK: %[[O0:.*]], %[[O1:.*]] = ortumcore.cx_bfly %[[A]], %[[T]]
// CHECK-SAME: shift = 1
// CHECK-SAME: variant = #ortumcore<cx_bfly_variant plain>
// CHECK: return %[[O0]], %[[O1]]
// CHECK-NOT: ondsp.cx_butterfly

func.func @selects_ntp_sat_shift0(%a: i32, %b: i32) -> (i32, i32) {
  %tw = arith.constant 1518511486 : i32  // 0x5A82A57E
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_ties_positive, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 0, rounding = nearest_ties_positive, overflow = saturate, saturate_to = i16>
  } : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

// CHECK-LABEL: func.func @selects_ntp_sat_shift0(
// CHECK: ortumcore.cx_mul_conj
// CHECK-SAME: rounding = #ortumcore<cx_rounding nearest_ties_positive>
// CHECK: ortumcore.cx_bfly
// CHECK-SAME: shift = 0
// CHECK-NOT: ondsp.cx_butterfly

// The default nearest_even profile is not a target capability: untouched.
func.func @keeps_nearest_even(%a: i32, %b: i32) -> (i32, i32) {
  %tw = arith.constant 196610 : i32
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

// CHECK-LABEL: func.func @keeps_nearest_even(
// CHECK: ondsp.cx_butterfly
// CHECK-NOT: ortumcore.

// A runtime twiddle has no exact conjugation constant: untouched.
func.func @keeps_runtime_twiddle(%a: i32, %b: i32, %tw: i32) -> (i32, i32) {
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = toward_negative, overflow = wrap, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = wrap, saturate_to = i16>
  } : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

// CHECK-LABEL: func.func @keeps_runtime_twiddle(
// CHECK: ondsp.cx_butterfly
// CHECK-NOT: ortumcore.

// imag = -32768 has no representable exact conjugate: untouched.
func.func @keeps_unconjugatable_twiddle(%a: i32, %b: i32) -> (i32, i32) {
  %tw = arith.constant -2147483647 : i32  // 0x80000001
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = toward_negative, overflow = wrap, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = wrap, saturate_to = i16>
  } : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

// CHECK-LABEL: func.func @keeps_unconjugatable_twiddle(
// CHECK: ondsp.cx_butterfly
// CHECK-NOT: ortumcore.

// The cross combine consumes the swapped-layout product: real_hi + cross.
func.func @selects_cross(%a: i32, %b: i32) -> (i32, i32) {
  %tw = arith.constant 196610 : i32
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = toward_negative, overflow = wrap, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = wrap, saturate_to = i16>,
    variant = #ondsp.cx_butterfly_variant<cross>
  } : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

// CHECK-LABEL: func.func @selects_cross(
// CHECK: %[[CCONJ:.*]] = arith.constant -196606 : i32
// CHECK: ortumcore.cx_mul_conj
// CHECK-SAME: layout = #ortumcore<cx_layout real_hi>
// CHECK: ortumcore.cx_bfly
// CHECK-SAME: variant = #ortumcore<cx_bfly_variant cross>
// CHECK-NOT: ondsp.cx_butterfly

// A twiddle loaded from a compile-time table converts through a
// once-materialized elementwise-conjugated table shared by both loads.
func.func @selects_table_twiddle(%a: i32, %b: i32, %i: index, %j: index) -> (i32, i32, i32, i32) {
  %table = arith.constant dense<[2147418112, 196610]> : tensor<2xi32>  // 0x7FFF0000, imag 3 real 2
  %tw0 = tensor.extract %table[%i] : tensor<2xi32>
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw0 {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = toward_negative, overflow = wrap, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = wrap, saturate_to = i16>
  } : (i32, i32, i32) -> (i32, i32)
  %tw1 = tensor.extract %table[%j] : tensor<2xi32>
  %2, %3 = ondsp.cx_butterfly %a, %b, %tw1 {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = toward_negative, overflow = wrap, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = wrap, saturate_to = i16>,
    variant = #ondsp.cx_butterfly_variant<cross>
  } : (i32, i32, i32) -> (i32, i32)
  return %0, %1, %2, %3 : i32, i32, i32, i32
}

// CHECK-LABEL: func.func @selects_table_twiddle(
// CHECK: %[[CTBL:.*]] = arith.constant dense<[-2147418112, -196606]> : tensor<2xi32>
// CHECK-NOT: arith.constant dense<[-2147418112, -196606]>
// CHECK: %[[E0:.*]] = tensor.extract %[[CTBL]]
// CHECK: ortumcore.cx_mul_conj
// CHECK: ortumcore.cx_bfly
// CHECK: %[[E1:.*]] = tensor.extract %[[CTBL]]
// CHECK: ortumcore.cx_mul_conj
// CHECK-SAME: layout = #ortumcore<cx_layout real_hi>
// CHECK: ortumcore.cx_bfly
// CHECK-SAME: variant = #ortumcore<cx_bfly_variant cross>
// CHECK-NOT: ondsp.cx_butterfly

// A table holding the -j twiddle has no exact conjugate: untouched.
func.func @keeps_unconjugatable_table(%a: i32, %b: i32, %i: index) -> (i32, i32) {
  %table = arith.constant dense<[2147418112, -2147483648]> : tensor<2xi32>
  %tw = tensor.extract %table[%i] : tensor<2xi32>
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = toward_negative, overflow = wrap, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = wrap, saturate_to = i16>
  } : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

// CHECK-LABEL: func.func @keeps_unconjugatable_table(
// CHECK: ondsp.cx_butterfly
// CHECK-NOT: ortumcore.

// The exact unit variant has no product stage: b feeds the plain packed
// butterfly directly and no rotation is emitted at all.
func.func @selects_unit_without_rotation(%a: i32, %b: i32) -> (i32, i32) {
  %tw = arith.constant 32767 : i32
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = toward_negative, overflow = wrap, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = wrap, saturate_to = i16>,
    variant = #ondsp.cx_butterfly_variant<unit>
  } : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

// CHECK-LABEL: func.func @selects_unit_without_rotation(
// CHECK-SAME: %[[A:.*]]: i32, %[[B:.*]]: i32)
// CHECK-NOT: ortumcore.cx_mul_conj
// CHECK: %[[O0:.*]], %[[O1:.*]] = ortumcore.cx_bfly %[[A]], %[[B]]
// CHECK-SAME: variant = #ortumcore<cx_bfly_variant plain>
// CHECK-NOT: ortumcore.cx_mul_conj
// CHECK: return %[[O0]], %[[O1]]

// The exact unit cross swaps b's halves in plain arithmetic (the cross
// combine consumes its operand in the swapped packing) and emits no
// rotation either.
func.func @selects_unit_cross_swaps_halves(%a: i32, %b: i32) -> (i32, i32) {
  %tw = arith.constant 32767 : i32
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_ties_positive, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_ties_positive, overflow = saturate, saturate_to = i16>,
    variant = #ondsp.cx_butterfly_variant<unit_cross>
  } : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

// CHECK-LABEL: func.func @selects_unit_cross_swaps_halves(
// CHECK-SAME: %[[A:.*]]: i32, %[[B:.*]]: i32)
// CHECK-NOT: ortumcore.cx_mul_conj
// CHECK-DAG: %[[HI:.*]] = arith.shrui %[[B]], %{{.*}} : i32
// CHECK-DAG: %[[LO:.*]] = arith.shli %[[B]], %{{.*}} : i32
// CHECK: %[[SW:.*]] = arith.ori %[[LO]], %[[HI]] : i32
// CHECK: %[[O0:.*]], %[[O1:.*]] = ortumcore.cx_bfly %[[A]], %[[SW]]
// CHECK-SAME: variant = #ortumcore<cx_bfly_variant cross>
// CHECK-NOT: ortumcore.cx_mul_conj
// CHECK: return %[[O0]], %[[O1]]

// The reversed-carry index walk selects onto the 32-bit target progression
// through the top-aligned composition: shift both operands to the top,
// walk, and extract the width-bit result from the top.
func.func @selects_bitrev_walk(%cursor: index, %step: index) -> index {
  %next = ondsp.bitrev_add %cursor, %step {width = 5 : i64} : (index, index) -> index
  return %next : index
}

// CHECK-LABEL: func.func @selects_bitrev_walk(
// CHECK-SAME: %[[CUR:.*]]: index, %[[STEP:.*]]: index)
// CHECK-DAG: %[[AMT:.*]] = arith.constant 27 : i32
// CHECK-DAG: %[[CUR32:.*]] = arith.index_castui %[[CUR]] : index to i32
// CHECK-DAG: %[[STEP32:.*]] = arith.index_castui %[[STEP]] : index to i32
// CHECK-DAG: %[[CURTOP:.*]] = arith.shli %[[CUR32]], %[[AMT]] : i32
// CHECK-DAG: %[[STEPTOP:.*]] = arith.shli %[[STEP32]], %[[AMT]] : i32
// CHECK: %[[WALK:.*]] = ortumcore.bitrev_add %[[CURTOP]], %[[STEPTOP]]
// CHECK: %[[OUT:.*]] = arith.shrui %[[WALK]], %[[AMT]] : i32
// CHECK: %[[RES:.*]] = arith.index_castui %[[OUT]] : i32 to index
// CHECK-NOT: ondsp.bitrev_add
// CHECK: return %[[RES]]

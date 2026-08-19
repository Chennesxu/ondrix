// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s --check-prefix=STRUCTURE
// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s --check-prefix=TWIDDLE
// RUN: ondrix-opt %s --convert-ondrix-to-ondsp="vectorize-static-cfft" | FileCheck %s --check-prefix=VECTOR

func.func @cfft4_q31(%input: tensor<4xi64>) -> tensor<4xi64> {
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<forward>,
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>,
    product_scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 0, rounding = toward_negative, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  } : (tensor<4xi64>) -> tensor<4xi64>
  return %result : tensor<4xi64>
}

// STRUCTURE-LABEL: func.func @cfft4_q31
// STRUCTURE-NOT: ondrix.cfft
// STRUCTURE-COUNT-4: ondsp.cx_butterfly {{.*}} : (i64, i64, i64) -> (i64, i64)
// STRUCTURE: tensor.insert
// Frozen Q31 W(2,0) = +1 (saturated) and W(4,1) = -j.
// TWIDDLE-LABEL: func.func @cfft4_q31
// TWIDDLE-DAG: arith.constant 2147483647 : i64
// TWIDDLE-DAG: arith.constant -9223372036854775808 : i64

func.func @cfft8_q31(%input: tensor<8xi64>) -> tensor<8xi64> {
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<forward>,
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>,
    product_scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 0, rounding = toward_negative, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  } : (tensor<8xi64>) -> tensor<8xi64>
  return %result : tensor<8xi64>
}

// STRUCTURE-LABEL: func.func @cfft8_q31
// STRUCTURE-NOT: ondrix.cfft
// STRUCTURE-COUNT-12: ondsp.cx_butterfly {{.*}} : (i64, i64, i64) -> (i64, i64)
// The frozen offline table: W(8,1) = 1518500250 - j1518500250 and
// W(8,3) = -1518500250 - j1518500250, packed imaginary-high.
// TWIDDLE-LABEL: func.func @cfft8_q31
// TWIDDLE-DAG: arith.constant -6521908911199323750 : i64
// TWIDDLE-DAG: arith.constant -6521908909941356954 : i64

// The Vector-batched mode follows the packed profile, so the Q31 combine
// stages batch over the i64 container exactly as Q15 does over i32.
// VECTOR-LABEL: func.func @cfft8_q31
// VECTOR: ondsp.cx_butterfly {{.*}} : (vector<2xi64>, vector<2xi64>, vector<2xi64>) -> (vector<2xi64>, vector<2xi64>)
// VECTOR: ondsp.cx_butterfly {{.*}} : (vector<4xi64>, vector<4xi64>, vector<4xi64>) -> (vector<4xi64>, vector<4xi64>)
// VECTOR: vector.extract

func.func @icfft8_q31(%input: tensor<8xi64>) -> tensor<8xi64> {
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<inverse>,
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>,
    product_scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 0, rounding = toward_negative, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  } : (tensor<8xi64>) -> tensor<8xi64>
  return %result : tensor<8xi64>
}

// STRUCTURE-LABEL: func.func @icfft8_q31
// STRUCTURE-COUNT-12: ondsp.cx_butterfly {{.*}} : (i64, i64, i64) -> (i64, i64)
// The inverse table is stored separately, not negated: W(4,1) = +j saturates
// to 2147483647 while its forward counterpart is exactly -2147483648.
// TWIDDLE-LABEL: func.func @icfft8_q31
// TWIDDLE-DAG: arith.constant 9223372032559808512 : i64
// TWIDDLE-DAG: arith.constant 6521908914236324250 : i64
// TWIDDLE-DAG: arith.constant 6521908915494291046 : i64

// The maximum supported Q31 extent, which is also the only place the deeper
// frozen table rows are reachable. Without these pins nothing outside the
// generator script would notice a corrupted N=32/64 entry.
func.func @cfft64_q31(%input: tensor<64xi64>) -> tensor<64xi64> {
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<forward>,
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>,
    product_scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 0, rounding = toward_negative, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  } : (tensor<64xi64>) -> tensor<64xi64>
  return %result : tensor<64xi64>
}

// STRUCTURE-LABEL: func.func @cfft64_q31
// STRUCTURE-NOT: ondrix.cfft
// STRUCTURE-COUNT-192: ondsp.cx_butterfly {{.*}} : (i64, i64, i64) -> (i64, i64)
// W(64,1) = 2137142927 - j210490206, W(64,21) = -1012316784 - j1893911494,
// and W(64,31) = -2137142927 - j210490206 come from stage-64 rows the smaller
// extents never touch.
// TWIDDLE-LABEL: func.func @cfft64_q31
// TWIDDLE-DAG: arith.constant -904048548761160049 : i64
// TWIDDLE-DAG: arith.constant -8134287924965849712 : i64
// TWIDDLE-DAG: arith.constant -904048548740478607 : i64

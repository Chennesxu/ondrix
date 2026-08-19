// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s
// RUN: ondrix-opt %s --convert-ondrix-to-ondsp="vectorize-static-cfft" | FileCheck %s --check-prefix=VECTOR
// RUN: ondrix-opt %s --convert-ondrix-to-ondsp="fft-loops" | FileCheck %s --check-prefix=LOOPS

func.func @rfft8_q31(%input: tensor<8xi32>) -> tensor<5xi64> {
  %result = ondrix.rfft %input {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>,
    product_scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 0, rounding = toward_negative, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  } : (tensor<8xi32>) -> tensor<5xi64>
  return %result : tensor<5xi64>
}

// The real samples pack into the i64 container, the DC and Nyquist bins
// canonicalize through the storage width, and the compact result keeps
// N/2 + 1 bins.
// CHECK-LABEL: func.func @rfft8_q31
// CHECK-NOT: ondrix.rfft
// CHECK: arith.extui {{.*}} : i32 to i64
// CHECK-COUNT-12: ondsp.cx_butterfly {{.*}} : (i64, i64, i64) -> (i64, i64)
// CHECK: arith.trunci {{.*}} : i64 to i32
// CHECK-NEXT: arith.extui {{.*}} : i32 to i64
// CHECK: tensor.insert {{.*}} : tensor<5xi64>

// VECTOR-LABEL: func.func @rfft8_q31
// VECTOR: ondsp.cx_butterfly {{.*}} : (vector<2xi64>, vector<2xi64>, vector<2xi64>) -> (vector<2xi64>, vector<2xi64>)
// VECTOR: ondsp.cx_butterfly {{.*}} : (vector<4xi64>, vector<4xi64>, vector<4xi64>) -> (vector<4xi64>, vector<4xi64>)

// LOOPS-LABEL: func.func @rfft8_q31
// LOOPS-NOT: ondrix.rfft
// LOOPS: scf.for
// LOOPS: arith.extui {{.*}} : i32 to i64
// LOOPS: ondsp.cx_butterfly {{.*}} : (i64, i64, i64) -> (i64, i64)
// LOOPS: tensor.extract_slice {{.*}} : tensor<8xi64> to tensor<5xi64>

func.func @irfft8_q31(%input: tensor<5xi64>) -> tensor<8xi32> {
  %result = ondrix.irfft %input {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>,
    product_scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 0, rounding = toward_negative, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  } : (tensor<5xi64>) -> tensor<8xi32>
  return %result : tensor<8xi32>
}

// The Hermitian mirror conjugates through the 32-bit rails: negation of the
// storage minimum saturates rather than wrapping.
// CHECK-LABEL: func.func @irfft8_q31
// CHECK-NOT: ondrix.irfft
// CHECK: arith.cmpi eq, {{.*}}, %c-2147483648_i32
// CHECK: arith.select
// CHECK-COUNT-12: ondsp.cx_butterfly {{.*}} : (i64, i64, i64) -> (i64, i64)
// CHECK: arith.trunci {{.*}} : i64 to i32
// CHECK: tensor.insert {{.*}} : tensor<8xi32>

// RUN: ondrix-opt %s --convert-ondsp-fixed-to-scalar --convert-vector-to-llvm --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir --implicit-check-not=ondsp.
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/cx_butterfly_q31_aot.c %t.o -o %t
// RUN: %t

// Operation-level differential gate for the packed-Q31 butterfly. This is
// where the exactness of the product carrier is actually exercised:
// ondsp.cx_butterfly accepts an arbitrary SSA twiddle, so b = w =
// (-2^31, -2^31) drives the imaginary cross sum br*wi + bi*wr to exactly
// 2^63, one past INT64_MAX. That corner refutes a WRAPPING i64 carrier — it
// wraps to -2^63 and the requantization saturates to the opposite rail. It
// does not make i128 the only legal implementation: 2^63 is the single value
// past i64, and an overflow-aware saturating i64 sum would round and saturate
// identically under this profile's shift-31 nearest-even boundary, so such an
// alternative would need its own equivalence proof, not this gate. The CFFT
// gate cannot reach the corner at all, because its twiddles are frozen
// unit-circle constants that bound the same sum by 0.7071 * INT64_MAX.
// CHECK-NOT: ondrix.

func.func @cx_butterfly_q31_result(
    %a: i64, %b: i64, %twiddle: i64, %result_index: i32) -> i64 {
  %out0, %out1 = ondsp.cx_butterfly %a, %b, %twiddle {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 31, rounding = nearest_even, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i32>
  } : (i64, i64, i64) -> (i64, i64)
  %zero = arith.constant 0 : i32
  %select_out0 = arith.cmpi eq, %result_index, %zero : i32
  %result = arith.select %select_out0, %out0, %out1 : i64
  return %result : i64
}

// The other admitted Q31 product selection, the scalar target's raw-high term.
// Its whole point is that the four cross terms floor independently, so the
// harness carries a fused-floor alternative and refuses to run unless the
// named discriminator separates the two.
func.func @cx_butterfly_q31_raw_high_result(
    %a: i64, %b: i64, %twiddle: i64, %result_index: i32) -> i64 {
  %out0, %out1 = ondsp.cx_butterfly %a, %b, %twiddle {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>,
    product_scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 0, rounding = toward_negative, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  } : (i64, i64, i64) -> (i64, i64)
  %zero = arith.constant 0 : i32
  %select_out0 = arith.cmpi eq, %result_index, %zero : i32
  %result = arith.select %select_out0, %out0, %out1 : i64
  return %result : i64
}

// The verifier admits the same profile elementwise over a fixed Vector
// carrier, so that admitted surface is gated here as well rather than left
// verifier-only. Each lane must be a complete independent butterfly: the
// harness pairs the carrier-refutation corner with a tie case in one
// operation, in both lane orders, so a lowering that shared a carrier across
// lanes or swapped lanes would fail on directed data.
func.func @cx_butterfly_q31_vector_result(
    %a0: i64, %a1: i64, %b0: i64, %b1: i64, %w0: i64, %w1: i64,
    %result_index: i32, %lane: i32) -> i64 {
  %seed = arith.constant dense<0> : vector<2xi64>
  %a_low = vector.insert %a0, %seed [0] : i64 into vector<2xi64>
  %a = vector.insert %a1, %a_low [1] : i64 into vector<2xi64>
  %b_low = vector.insert %b0, %seed [0] : i64 into vector<2xi64>
  %b = vector.insert %b1, %b_low [1] : i64 into vector<2xi64>
  %w_low = vector.insert %w0, %seed [0] : i64 into vector<2xi64>
  %w = vector.insert %w1, %w_low [1] : i64 into vector<2xi64>
  %out0, %out1 = ondsp.cx_butterfly %a, %b, %w {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 31, rounding = nearest_even, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i32>
  } : (vector<2xi64>, vector<2xi64>, vector<2xi64>) -> (vector<2xi64>, vector<2xi64>)
  %zero = arith.constant 0 : i32
  %select_out0 = arith.cmpi eq, %result_index, %zero : i32
  %chosen = arith.select %select_out0, %out0, %out1 : vector<2xi64>
  %result = vector.extractelement %chosen[%lane : i32] : vector<2xi64>
  return %result : i64
}

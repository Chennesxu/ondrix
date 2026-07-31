// RUN: ondrix-opt %s --convert-ondsp-fixed-to-scalar --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir --implicit-check-not=ondsp.
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/cx_butterfly_q31_aot.c %t.o -o %t
// RUN: %t

// Operation-level differential gate for the packed-Q31 butterfly. This is
// where the i128 product carrier is actually witnessed: ondsp.cx_butterfly
// accepts an arbitrary SSA twiddle, so b = w = (-2^31, -2^31) drives the
// imaginary cross sum br*wi + bi*wr to exactly 2^63, one past INT64_MAX. The
// CFFT gate cannot witness this, because its twiddles are frozen unit-circle
// constants that bound the same sum by 0.7071 * INT64_MAX.
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

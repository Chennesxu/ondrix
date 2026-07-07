// RUN: ondrix-opt %s --convert-ondsp-to-ortumcore | FileCheck %s

func.func @trivial_butterfly(%a: i32, %b: i32, %tw: i32) -> (i32, i32) {
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>, numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, trivial_twiddle = true} : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

// CHECK-LABEL: func.func @trivial_butterfly
// CHECK: ortumcore.fft_trivial_stage
// CHECK-SAME: variant = 7
// CHECK-NOT: ondsp.cx_butterfly

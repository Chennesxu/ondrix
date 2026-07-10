// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s

func.func @butterfly_q15(%a: i32, %b: i32, %tw: i32) -> (i32, i32) {
  %0, %1 = ondrix.butterfly %a, %b, %tw {layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>, numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

// CHECK-LABEL: func.func @butterfly_q15
// CHECK-NOT: ondrix.butterfly
// CHECK: ondsp.cx_butterfly
// CHECK: #ondsp.cx_layout<packed_i16_imag_hi_real_lo>
// CHECK: #ondsp.product<full>

// RUN: not ondrix-opt %s --convert-ondsp-to-ortumcore 2>&1 | FileCheck %s

func.func @bad_butterfly_storage(%a: i16, %b: i16, %tw: i16) -> (i16, i16) {
  // CHECK: packed q15 butterfly operands must use i32 storage
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>, numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (i16, i16, i16) -> (i16, i16)
  return %0, %1 : i16, i16
}

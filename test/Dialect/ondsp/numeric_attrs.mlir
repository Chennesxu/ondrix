// RUN: ondrix-opt %s | FileCheck %s

// CHECK-LABEL: func.func @numeric_attrs
func.func @numeric_attrs(%x: i16, %y: f32) -> (i16, f32) {
  // CHECK: #ondsp.fixed<signed, storage = i16, frac = 15>
  %0 = ondsp.assume_numeric %x {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>} : (i16) -> i16

  // CHECK: #ondsp.fixed<unsigned, storage = i16, frac = 15>
  %1 = ondsp.assume_numeric %0 {numeric = #ondsp.fixed<unsigned, storage = i16, frac = 15>} : (i16) -> i16

  // CHECK: #ondsp.fp<format = f32, contract = fma>
  %2 = ondsp.assume_numeric %y {numeric = #ondsp.fp<format = f32, contract = fma>} : (f32) -> f32

  // CHECK: #ondsp.scale<shift_right = 15, shift_left = 0, order = shift_round_saturate>
  %3 = ondsp.round_shift %1 {scale = #ondsp.scale<shift_right = 15, shift_left = 0, order = shift_round_saturate>} : (i16) -> i16
  return %3, %2 : i16, f32
}

// RUN: ondrix-compile %S/Inputs/q15_fir_decimate.ox | FileCheck %s

// CHECK-LABEL: func.func @q15_fir_decimate(
// CHECK-SAME: tensor<12xi16>
// CHECK-SAME: tensor<5xi16>
// CHECK-SAME: tensor<4xi16>
// CHECK: tensor.empty() : tensor<4xi16>
// CHECK: ondrix.fir_decimate
// CHECK-SAME: accumulator = !ondsp.acc<storage = i34, frac = 30, signed, update_overflow = wrap>
// CHECK-SAME: factor = 2
// CHECK-SAME: numeric = #ondsp.fixed<signed, storage = i16, frac = 15>

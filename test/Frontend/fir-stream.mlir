// RUN: ondrix-compile %S/Inputs/q15_fir_stream.ox | FileCheck %s

// CHECK-LABEL: func.func @q15_fir_stream(
// CHECK-SAME: tensor<?xi16>
// CHECK-SAME: tensor<3xi16>
// CHECK-SAME: tensor<2xi16>
// CHECK-SAME: -> (tensor<?xi16>, tensor<2xi16>)
// CHECK: %[[OUTPUT:.*]], %[[NEXT:.*]] = ondrix.fir_stream
// CHECK-SAME: accumulator = !ondsp.acc<storage = i33, frac = 30, signed, update_overflow = wrap>
// CHECK-SAME: overflow = #ondsp.overflow<saturate>
// CHECK-SAME: rounding = #ondsp.rounding<nearest_even>
// CHECK: return %[[OUTPUT]], %[[NEXT]]

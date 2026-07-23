// RUN: ondrix-compile %S/Inputs/q15_fir_filter_valid.ox | FileCheck %s --check-prefix=Q15
// RUN: ondrix-compile %S/Inputs/q31_fir_filter_valid.ox | FileCheck %s --check-prefix=Q31
// RUN: ondrix-compile %S/Inputs/f32_fir_filter_valid.ox | FileCheck %s --check-prefix=F32

// Q15-LABEL: func.func @q15_fir_filter_valid(
// Q15-SAME: tensor<?xi16>
// Q15-SAME: tensor<?xi16>
// Q15-SAME: -> tensor<?xi16>
// Q15: %[[INPUT_LEN:.*]] = tensor.dim
// Q15: %[[COEFF_LEN:.*]] = tensor.dim
// Q15: arith.cmpi ugt, %[[COEFF_LEN]]
// Q15: arith.cmpi uge, %[[INPUT_LEN]], %[[COEFF_LEN]]
// Q15: %[[INIT:.*]] = tensor.empty
// Q15: ondrix.fir_filter
// Q15-SAME: %[[INIT]]
// Q15-SAME: accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
// Q15-SAME: boundary = #ondrix.fir_boundary<valid>
// Q15-SAME: rounding = #ondsp.rounding<nearest_even>

// Q31-LABEL: func.func @q31_fir_filter_valid(
// Q31-SAME: tensor<6xi32>
// Q31-SAME: tensor<3xi32>
// Q31-SAME: -> tensor<4xi32>
// Q31: tensor.empty() : tensor<4xi32>
// Q31: ondrix.fir_filter
// Q31-SAME: accumulator = !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>
// Q31-SAME: overflow = #ondsp.overflow<wrap>

// F32-LABEL: func.func @f32_fir_filter_valid(
// F32-SAME: tensor<?xf32>
// F32-SAME: -> tensor<?xf32>
// F32: ondrix.fir_filter
// F32-SAME: numeric = #ondsp.fp<format = f32, contract = fma>
// F32-NOT: accumulator =

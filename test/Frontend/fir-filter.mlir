// RUN: ondrix-compile %S/Inputs/q15_fir_filter_valid.ox | FileCheck %s --check-prefix=Q15
// RUN: ondrix-compile %S/Inputs/q31_fir_filter_valid.ox | FileCheck %s --check-prefix=Q31
// RUN: ondrix-compile %S/Inputs/f32_fir_filter_valid.ox | FileCheck %s --check-prefix=F32
// RUN: ondrix-compile %S/Inputs/q15_fir_filter_constexpr.ox | FileCheck %s --check-prefix=CONST
// RUN: ondrix-compile %S/Inputs/q15_fir_filter_constexpr.ox | ondrix-opt --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --cse --canonicalize --vectorize-ondsp-constant-saturating-memref-reduce="vector-width=4 max-elements=64" --normalize-ondsp-fixed-vector-reduce | FileCheck %s --check-prefix=PROVEN
// RUN: ondrix-compile %S/Inputs/q15_fir_filter_full.ox | FileCheck %s --check-prefix=FULL-Q15
// RUN: ondrix-compile %S/Inputs/q31_fir_filter_full.ox | FileCheck %s --check-prefix=FULL-Q31
// RUN: ondrix-compile %S/Inputs/f32_fir_filter_full.ox | FileCheck %s --check-prefix=FULL-F32

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

// CONST: memref.global "private" constant @__ox_q15_fir_filter_constexpr_coefficients
// CONST-LABEL: func.func @q15_fir_filter_constexpr(
// CONST-SAME: tensor<8xi16>) -> tensor<5xi16>
// CONST: memref.get_global @__ox_q15_fir_filter_constexpr_coefficients
// CONST: bufferization.to_tensor {{.*}} restrict
// CONST: ondrix.fir_filter

// PROVEN-LABEL: func.func @q15_fir_filter_constexpr(
// PROVEN-SAME: memref<8xi16>) -> memref<5xi16>
// PROVEN: memref.get_global @__ox_q15_fir_filter_constexpr_coefficients
// PROVEN: vector.reduction <add>, {{.*}} : vector<4xi64> into i64
// PROVEN: ondsp.acc_add_term
// PROVEN-NOT: ondsp.reduce_mac

// FULL-Q15-LABEL: func.func @q15_fir_filter_full(
// FULL-Q15-SAME: tensor<6xi16>
// FULL-Q15-SAME: tensor<3xi16>
// FULL-Q15-SAME: -> tensor<8xi16>
// FULL-Q15: ondrix.fir_filter
// FULL-Q15-SAME: boundary = #ondrix.fir_boundary<full>

// FULL-Q31-LABEL: func.func @q31_fir_filter_full(
// FULL-Q31-SAME: -> tensor<6xi32>
// FULL-Q31: ondrix.fir_filter
// FULL-Q31-SAME: update_overflow = wrap
// FULL-Q31-SAME: boundary = #ondrix.fir_boundary<full>

// FULL-F32-LABEL: func.func @f32_fir_filter_full(
// FULL-F32-SAME: -> tensor<7xf32>
// FULL-F32: boundary = #ondrix.fir_boundary<full>
// FULL-F32-SAME: numeric = #ondsp.fp<format = f32, contract = fma>

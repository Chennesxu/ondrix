// RUN: ondrix-compile %S/Inputs/q15_rms.ox | FileCheck %s
// RUN: ondrix-compile %S/Inputs/q15_rms_floor.ox | FileCheck %s --check-prefix=FLOOR
// RUN: not ondrix-compile %S/Inputs/invalid_rms_extent.ox 2>&1 | FileCheck %s --check-prefix=EXTENT
// RUN: not ondrix-compile %S/Inputs/invalid_rms_result.ox 2>&1 | FileCheck %s --check-prefix=SINGLETON
// RUN: not ondrix-compile %S/Inputs/invalid_rms_rounding.ox 2>&1 | FileCheck %s --check-prefix=ROUNDING
// RUN: not ondrix-compile %S/Inputs/invalid_rms_ties_positive.ox 2>&1 | FileCheck %s --check-prefix=TIES

// CHECK-LABEL: func.func @q15_rms(
// CHECK-SAME: %[[INPUT:.*]]: tensor<64xi16>) -> tensor<1xi16>
// CHECK: %[[RESULT:.*]] = ondrix.rms %[[INPUT]]
// CHECK-SAME: numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
// CHECK-SAME: rounding = #ondsp.rounding<nearest_even>
// CHECK: return %[[RESULT]] : tensor<1xi16>

// The declared root rounding routes to the op attribute; the mean boundary
// stays nearest even inside the contract regardless of this choice.
// FLOOR-LABEL: func.func @q15_rms_floor(
// FLOOR: ondrix.rms
// FLOOR-SAME: rounding = #ondsp.rounding<toward_negative>

// EXTENT: invalid_rms_extent.ox:2:10: error: rms currently requires a power-of-two input extent in [2, 4096]
// SINGLETON: invalid_rms_result.ox:2:10: error: rms returns a single-element tensor
// ROUNDING: invalid_rms_rounding.ox:2:10: error: rms root_rounding must be nearest_even or toward_negative
// A newly declared dialect rounding mode does not widen a binding: rms
// admits only the modes its own contract admits.
// TIES: invalid_rms_ties_positive.ox:2:10: error: rms root_rounding must be nearest_even or toward_negative

// RUN: ondrix-compile %S/Inputs/f32_rms.ox | FileCheck %s --check-prefix=F32

// The f32 binding names a contract instead of a root rounding mode, and its
// extent need not be a power of two.
// F32-LABEL: func.func @f32_rms
// F32-SAME: tensor<10xf32>) -> tensor<1xf32>
// F32: ondrix.rms
// F32-SAME: numeric = #ondsp.fp<format = f32, contract = off>

// RUN: ondrix-compile %S/Inputs/f32_rms_scalar.ox | FileCheck %s --check-prefix=SCALAR
// RUN: not ondrix-compile %S/Inputs/invalid_rms_scalar_q15.ox 2>&1 | FileCheck %s --check-prefix=SCALARQ15
// RUN: not ondrix-compile %S/Inputs/invalid_rms_scalar_tensor.ox 2>&1 | FileCheck %s --check-prefix=SCALARBUF

// The scalar spelling rides the dot route with the same buffer as both
// operands; pinning both discriminates a two-operand dot substitution.
// SCALAR-LABEL: func.func @f32_rms_scalar(
// SCALAR-SAME: %[[X:.*]]: memref<10xf32>) -> f32
// SCALAR: %[[SS:.*]] = ondrix.dot %[[X]], %[[X]] {numeric = #ondsp.fp<format = f32, contract = fast>}
// SCALAR: %[[N:.*]] = arith.constant 1.000000e+01 : f32
// SCALAR: %[[MEAN:.*]] = arith.divf %[[SS]], %[[N]] : f32
// SCALAR: %[[ROOT:.*]] = math.sqrt %[[MEAN]] : f32
// SCALAR: return %[[ROOT]] : f32

// SCALARQ15: invalid_rms_scalar_q15.ox:2:10: error: scalar rms requires one f32 buffer operand
// SCALARBUF: invalid_rms_scalar_tensor.ox:2:10: error: scalar rms requires one f32 buffer operand

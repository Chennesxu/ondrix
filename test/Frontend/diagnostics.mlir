// RUN: not ondrix-compile %S/Inputs/invalid_kernel_keyword.ox 2>&1 | FileCheck %s --check-prefix=KERNEL
// RUN: not ondrix-compile %S/Inputs/invalid_accumulator_width.ox 2>&1 | FileCheck %s --check-prefix=WIDTH
// RUN: not ondrix-compile %S/Inputs/unknown_operand.ox 2>&1 | FileCheck %s --check-prefix=OPERAND
// RUN: not ondrix-compile %S/Inputs/invalid_mixed_types.ox 2>&1 | FileCheck %s --check-prefix=MIXED
// RUN: not ondrix-compile %S/Inputs/invalid_constexpr_dynamic_window.ox 2>&1 | FileCheck %s --check-prefix=CONSTEXPR-DYNAMIC
// RUN: not ondrix-compile %S/Inputs/invalid_constexpr_lhs.ox 2>&1 | FileCheck %s --check-prefix=CONSTEXPR-LHS
// RUN: not ondrix-compile %S/Inputs/invalid_constexpr_q15_range.ox 2>&1 | FileCheck %s --check-prefix=CONSTEXPR-RANGE
// RUN: not ondrix-compile %S/Inputs/invalid_q31_accumulator_width.ox 2>&1 | FileCheck %s --check-prefix=Q31-WIDTH
// RUN: not ondrix-compile %S/Inputs/invalid_constexpr_q31_range.ox 2>&1 | FileCheck %s --check-prefix=Q31-RANGE
// RUN: not ondrix-compile %S/Inputs/invalid_fir_filter_buffer.ox 2>&1 | FileCheck %s --check-prefix=FILTER-BUFFER
// RUN: not ondrix-compile %S/Inputs/invalid_fir_filter_boundary.ox 2>&1 | FileCheck %s --check-prefix=FILTER-BOUNDARY
// RUN: not ondrix-compile %S/Inputs/invalid_tensor_dot.ox 2>&1 | FileCheck %s --check-prefix=TENSOR-DOT
// RUN: not ondrix-compile %S/Inputs/invalid_fir_filter_full_dynamic.ox 2>&1 | FileCheck %s --check-prefix=FULL-DYNAMIC
// RUN: not ondrix-compile %S/Inputs/invalid_fir_filter_full_shape.ox 2>&1 | FileCheck %s --check-prefix=FULL-SHAPE
// RUN: not ondrix-compile %S/Inputs/invalid_fir_filter_full_overflow.ox 2>&1 | FileCheck %s --check-prefix=FULL-OVERFLOW
// RUN: not ondrix-compile %S/Inputs/invalid_fir_filter_empty_constexpr.ox 2>&1 | FileCheck %s --check-prefix=EMPTY-CONSTEXPR
// RUN: not ondrix-compile %S/Inputs/invalid_fir_filter_valid_mixed_shape.ox 2>&1 | FileCheck %s --check-prefix=VALID-MIXED-SHAPE
// RUN: not ondrix-compile %S/Inputs/invalid_butterfly_result.ox 2>&1 | FileCheck %s --check-prefix=BUTTERFLY-RESULT

// KERNEL: invalid_kernel_keyword.ox:1:1: error: unsupported top-level construct 'kernel'; expected 'def'
// KERNEL-NEXT: kernel q15_dot(lhs: buffer[q15], rhs: buffer[q15]) -> q15:
// KERNEL-NEXT: ^

// WIDTH: invalid_accumulator_width.ox:2:10: error: the executable Q15 profile requires exact accumulator width 40
// WIDTH-NEXT: return dot(lhs, rhs,

// OPERAND: unknown_operand.ox:2:10: error: unknown reduction operand 'coefficients'
// OPERAND-NEXT: return dot(lhs, coefficients,

// MIXED: invalid_mixed_types.ox:1:29: error: parameter element types must match the kernel result type

// CONSTEXPR-DYNAMIC: invalid_constexpr_dynamic_window.ox:2:5: error: a constexpr reduction operand requires a static left operand extent

// CONSTEXPR-LHS: invalid_constexpr_lhs.ox:3:10: error: constexpr is supported only for the right operand of a fixed-point reduction

// CONSTEXPR-RANGE: invalid_constexpr_q15_range.ox:2:29: error: Q15 constexpr coefficient is outside signed i16 storage range

// Q31-WIDTH: invalid_q31_accumulator_width.ox:2:10: error: the executable Q31 profile requires exact accumulator width 64

// Q31-RANGE: invalid_constexpr_q31_range.ox:2:29: error: Q31 constexpr coefficient is outside signed i32 storage range

// FILTER-BUFFER: invalid_fir_filter_buffer.ox:2:10: error: fir_filter currently requires tensor input and tensor or constexpr coefficients

// FILTER-BOUNDARY: invalid_fir_filter_boundary.ox:2:10: error: fir_filter supports only boundary=valid or boundary=full

// TENSOR-DOT: invalid_tensor_dot.ox:2:10: error: scalar dot and fir currently require buffer operands

// FULL-DYNAMIC: invalid_fir_filter_full_dynamic.ox:2:10: error: full fir_filter currently requires static input, coefficient, and result extents

// FULL-SHAPE: invalid_fir_filter_full_shape.ox:2:10: error: static fir_filter result extent does not match full convolution

// FULL-OVERFLOW: invalid_fir_filter_full_overflow.ox:2:10: error: full fir_filter result extent overflows index

// EMPTY-CONSTEXPR: invalid_fir_filter_empty_constexpr.ox:3:5: error: constexpr reduction operand cannot be empty

// VALID-MIXED-SHAPE: invalid_fir_filter_valid_mixed_shape.ox:2:10: error: a static valid fir_filter result requires static input and coefficient extents

// BUTTERFLY-RESULT: invalid_butterfly_result.ox:3:10: error: butterfly requires three complex_q15 scalar parameters and two complex_q15 results

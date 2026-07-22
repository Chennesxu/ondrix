// RUN: not ondrix-compile %S/Inputs/invalid_python_def.ox 2>&1 | FileCheck %s --check-prefix=PYTHON
// RUN: not ondrix-compile %S/Inputs/invalid_accumulator_width.ox 2>&1 | FileCheck %s --check-prefix=WIDTH
// RUN: not ondrix-compile %S/Inputs/unknown_operand.ox 2>&1 | FileCheck %s --check-prefix=OPERAND
// RUN: not ondrix-compile %S/Inputs/invalid_mixed_types.ox 2>&1 | FileCheck %s --check-prefix=MIXED
// RUN: not ondrix-compile %S/Inputs/invalid_f32_fir.ox 2>&1 | FileCheck %s --check-prefix=F32-FIR

// PYTHON: invalid_python_def.ox:1:1: error: unsupported top-level construct 'def'; expected 'kernel'
// PYTHON-NEXT: def q15_dot(lhs, rhs):
// PYTHON-NEXT: ^

// WIDTH: invalid_accumulator_width.ox:2:10: error: the executable Q15 profile requires exact accumulator width 40
// WIDTH-NEXT: return dot(lhs, rhs,

// OPERAND: unknown_operand.ox:2:10: error: unknown reduction operand 'coefficients'
// OPERAND-NEXT: return dot(lhs, coefficients,

// MIXED: invalid_mixed_types.ox:1:32: error: parameter element types must match the kernel result type

// F32-FIR: invalid_f32_fir.ox:2:10: error: the current f32 frontend slice supports dot but not FIR

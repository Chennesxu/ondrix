// RUN: not ondrixc %S/Inputs/invalid_python_def.ox 2>&1 | FileCheck %s --check-prefix=PYTHON
// RUN: not ondrixc %S/Inputs/invalid_accumulator_width.ox 2>&1 | FileCheck %s --check-prefix=WIDTH
// RUN: not ondrixc %S/Inputs/unknown_operand.ox 2>&1 | FileCheck %s --check-prefix=OPERAND

// PYTHON: invalid_python_def.ox:1:1: error: unsupported top-level construct 'def'; expected 'kernel'
// PYTHON-NEXT: def q15_dot(lhs, rhs):
// PYTHON-NEXT: ^

// WIDTH: invalid_accumulator_width.ox:2:10: error: the executable Q15 profile requires exact accumulator width 40
// WIDTH-NEXT: return dot(lhs, rhs,

// OPERAND: unknown_operand.ox:2:10: error: unknown dot operand 'coefficients'
// OPERAND-NEXT: return dot(lhs, coefficients,

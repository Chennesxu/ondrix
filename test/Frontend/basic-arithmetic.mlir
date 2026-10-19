// RUN: ondrix-compile %S/Inputs/scalar_arithmetic.ox | FileCheck %s --check-prefix=INT
// RUN: ondrix-compile %S/Inputs/scalar_predicate.ox | FileCheck %s --check-prefix=BOOL
// RUN: ondrix-compile %S/Inputs/f32_arithmetic.ox | FileCheck %s --check-prefix=FLOAT
// RUN: ondrix-compile %S/Inputs/q15_unary_minus.ox | FileCheck %s --check-prefix=FIXED
// INT-LABEL: func.func @scalar_arithmetic
// INT: arith.muli
// INT: arith.addi
// INT: cf.assert
// INT: arith.divsi
// BOOL-LABEL: func.func @scalar_predicate
// BOOL: scf.if
// BOOL: scf.if
// BOOL: arith.divsi
// FLOAT-LABEL: func.func @f32_arithmetic
// FLOAT: arith.mulf
// FLOAT: arith.addf
// FLOAT: arith.negf
// FIXED: ondrix.negate
// FIXED-SAME: overflow = #ondsp.overflow<saturate>
// RUN: split-file %s %t
// RUN: ondrix-compile %t/scalar-constant-call.ox | FileCheck %s --check-prefix=CONSTANT
// RUN: ondrix-compile %t/scalar-builtin-local.ox | FileCheck %s --check-prefix=BUILTIN
// RUN: ondrix-compile %t/complex-readout.ox | FileCheck %s --check-prefix=READOUT
// RUN: ondrix-compile %t/call-context.ox | FileCheck %s --check-prefix=CONTEXT
// CONTEXT: arith.muli {{.*}} : i64
// CONTEXT: arith.addi {{.*}} : i64
// CONTEXT: arith.subi {{.*}} : i64
// CONTEXT: arith.subi {{.*}} : i64
// READOUT: ondrix.cx_power
// READOUT: arith.addf
// CONSTANT: arith.constant 4294967296 : i64
// CONSTANT: arith.cmpi eq
// BUILTIN: ondrix.dot
//--- scalar-constant-call.ox
def large() -> i64:
  return 4294967296
def compare_large() -> bool:
  return large() == 4294967296
//--- scalar-builtin-local.ox
def local_dot(x: buffer[f32,4], y: buffer[f32,4]) -> f32:
  a = x
  t = dot(a, y, contract=off)
  return t
//--- complex-readout.ox
def readout(x: tensor[complex_f32,4]) -> tensor[f32,4]:
  return power(x, contract=off) + 1
//--- call-context.ox
def call_context(x: i64) -> bool:
  return mult(2, 3) == x or add(2, 3) == x or sub(3, 2) == x or negate(2) == x

// RUN: split-file %s %t
// RUN: not ondrix-compile %t/mixed.ox 2>&1 | FileCheck %s --check-prefix=MIXED
// RUN: not ondrix-compile %t/logical.ox 2>&1 | FileCheck %s --check-prefix=LOGICAL
// RUN: not ondrix-compile %t/large.ox 2>&1 | FileCheck %s --check-prefix=LARGE
// RUN: not ondrix-compile %t/zero.ox 2>&1 | FileCheck %s --check-prefix=ZERO
// RUN: not ondrix-compile %t/overflow.ox 2>&1 | FileCheck %s --check-prefix=OVERFLOW
// RUN: not ondrix-compile %t/floatoverflow.ox 2>&1 | FileCheck %s --check-prefix=FLOATOVERFLOW
// RUN: not ondrix-compile %t/floatinvalid.ox 2>&1 | FileCheck %s --check-prefix=FLOATINVALID
// RUN: not ondrix-compile %t/shapes.ox 2>&1 | FileCheck %s --check-prefix=SHAPES
// RUN: not ondrix-compile %t/tensorcmp.ox 2>&1 | FileCheck %s --check-prefix=TENSORCMP
// RUN: not ondrix-compile %t/boolorder.ox 2>&1 | FileCheck %s --check-prefix=BOOLORDER
// RUN: not ondrix-compile %t/booltensor.ox 2>&1 | FileCheck %s --check-prefix=BOOLTENSOR
// RUN: not ondrix-compile %t/helper.ox 2>&1 | FileCheck %s --check-prefix=HELPER
// RUN: not ondrix-compile %t/helper-call.ox 2>&1 | FileCheck %s --check-prefix=HELPER
// HELPER: error: call argument does not match the declared parameter type
// MIXED: error: arithmetic operands must have the same element type
// LOGICAL: error: logical operators require scalar bool
// LARGE: error: integer literal is out of range
// RUN: not ondrix-compile %t/local-zero.ox 2>&1 | FileCheck %s --check-prefix=ZERO
// RUN: not ondrix-compile %t/helper-zero.ox 2>&1 | FileCheck %s --check-prefix=ZERO
// ZERO: error: integer division by zero
// OVERFLOW: error: integer division overflows
// FLOATOVERFLOW: error: f32 literal is out of range
// FLOATINVALID: error: invalid f32 literal
// SHAPES: error: arithmetic tensor operands must have matching extents
// TENSORCMP: error: comparisons require scalar operands
// BOOLORDER: error: bool only supports equality
// BOOLTENSOR: error: bool parameters must be scalar
//--- mixed.ox
def f(x: i32, y: i64) -> i32:
  return x + y

//--- logical.ox
def f(x: i32, y: bool) -> bool:
  return x and y

//--- large.ox
def f() -> i32:
  return 2147483648

//--- zero.ox
def f(x: i64) -> i64:
  return x / 0

//--- overflow.ox
def f() -> i64:
  return -9223372036854775808 / -1

//--- floatoverflow.ox
def f() -> f32:
  return 1e100

//--- floatinvalid.ox
def f() -> f32:
  return 1e+

//--- shapes.ox
def f(x: tensor[f32,4], y: tensor[f32,3]) -> tensor[f32,4]:
  return x + y

//--- tensorcmp.ox
def f(x: tensor[f32,4], y: tensor[f32,4]) -> bool:
  return x < y

//--- boolorder.ox
def f(x: bool, y: bool) -> bool:
  return x < y

//--- booltensor.ox
def f(x: tensor[bool,4]) -> bool:
  return x[0]

//--- helper.ox
def narrow_name(x: i32) -> i32:
  return x
def f(y: i64) -> i64:
  t = y + 1
  return (narrow_name(t)) + 1
//--- helper-call.ox
def h(x: i32) -> i32:
  return add(x, x)
def f(y: i64) -> i64:
  t = y + 1
  return h(t)
//--- local-zero.ox
def f(x: i64) -> i64:
  zero = 0
  return x / zero
//--- helper-zero.ox
def zero() -> i64:
  z = 0
  return z
def f(x: i64) -> i64:
  return x / zero()

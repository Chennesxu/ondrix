// RUN: split-file %s %t
// RUN: not ondrix-compile %t/readonly.ox 2>&1 | FileCheck %s --check-prefix=READONLY
// RUN: not ondrix-compile %t/retype.ox 2>&1 | FileCheck %s --check-prefix=RETYPE
// RUN: not ondrix-compile %t/scope.ox 2>&1 | FileCheck %s --check-prefix=SCOPE
// RUN: not ondrix-compile %t/carry-tensor.ox 2>&1 | FileCheck %s --check-prefix=CARRY
// RUN: not ondrix-compile %t/step-zero.ox 2>&1 | FileCheck %s --check-prefix=STEP
// RUN: not ondrix-compile %t/step-name.ox 2>&1 | FileCheck %s --check-prefix=STEP
// RUN: not ondrix-compile %t/step-range.ox 2>&1 | FileCheck %s --check-prefix=STEPRANGE
// RUN: not ondrix-compile %t/bounds.ox 2>&1 | FileCheck %s --check-prefix=BOUNDS
// RUN: not ondrix-compile %t/float-bounds.ox 2>&1 | FileCheck %s --check-prefix=BOUNDS
// RUN: not ondrix-compile %t/element.ox 2>&1 | FileCheck %s --check-prefix=LEN
// RUN: not ondrix-compile %t/subscript-index.ox 2>&1 | FileCheck %s --check-prefix=SUBSCRIPT
// RUN: not ondrix-compile %t/len-scalar.ox 2>&1 | FileCheck %s --check-prefix=LEN
// RUN: not ondrix-compile %t/complex-len.ox 2>&1 | FileCheck %s --check-prefix=LEN
// RUN: ondrix-compile %t/loop-name.ox | FileCheck %s --check-prefix=NAME
// NAME: return %arg0 : i32
// RUN: not ondrix-compile %t/index-mul.ox 2>&1 | FileCheck %s --check-prefix=INDEXMUL
// RUN: not ondrix-compile %t/index-mix.ox 2>&1 | FileCheck %s --check-prefix=INDEXMIX
// RUN: not ondrix-compile %t/index-constant.ox 2>&1 | FileCheck %s --check-prefix=INDEXCONST
// RUN: not ondrix-compile %t/indent.ox 2>&1 | FileCheck %s --check-prefix=INDENT
// RUN: not ondrix-compile %t/empty-body.ox 2>&1 | FileCheck %s --check-prefix=BODY
// RUN: not ondrix-compile %t/fixed.ox 2>&1 | FileCheck %s --check-prefix=FIXED
// READONLY: error: loop variable 'i' is read-only
// RETYPE: error: a reassignment must keep the local's type
// SCOPE: error: local 't' is only visible inside its loop body
// CARRY: error: a loop can only carry scalar locals
// STEP: error: range step must be a nonzero integer literal
// STEPRANGE: error: range step is out of range for the bound type
// BOUNDS: error: range bounds must share one index, i32 or i64 type
// SUBSCRIPT: error: an index must be an index, i32 or i64 scalar
// LEN: error: len and indexing currently require a rank-one f32 buffer or tensor parameter
// INDEXMUL: error: index arithmetic supports only + and -
// INDEXMIX: error: arithmetic operands must have the same element type
// INDEXCONST: error: constant index expression leaves the signed 32-bit range
// INDENT: error: inconsistent indentation in the loop body
// BODY: error: expected an indented loop body
// FIXED: error: loops require an ordinary i32, i64, bool or f32 function result
//--- readonly.ox
def f(n: i32) -> i32:
  s = 0
  for i in range(n):
    i = s
  return s

//--- retype.ox
def f(n: i32) -> i32:
  s = 0
  for i in range(n):
    s = 1.0
  return s

//--- scope.ox
def f(n: i32) -> i32:
  for i in range(n):
    t = i
  return t

//--- carry-tensor.ox
def f(x: tensor[f32,4], n: i32) -> tensor[f32,4]:
  acc = x
  for i in range(n):
    acc = acc * 0.5 + x
  return acc

//--- step-zero.ox
def f(n: i32) -> i32:
  s = 0
  for i in range(0, n, 0):
    s = s + i
  return s

//--- step-name.ox
def f(n: i32) -> i32:
  s = 0
  for i in range(0, n, n):
    s = s + i
  return s

//--- step-range.ox
def f(n: i32) -> i32:
  s = 0
  for i in range(0, n, 2147483648):
    s = s + i
  return s

//--- bounds.ox
def f(x: buffer[f32], n: i32) -> f32:
  s = 0.0
  for i in range(n, len(x)):
    s = s + x[i]
  return s

//--- float-bounds.ox
def f(n: f32) -> f32:
  s = 0.0
  for i in range(n):
    s = s + n
  return s

//--- element.ox
def f(x: buffer[q15]) -> f32:
  s = 0.0
  for i in range(len(x)):
    s = s + x[i]
  return s

//--- subscript-index.ox
def f(x: buffer[f32], k: f32) -> f32:
  return x[k]

//--- len-scalar.ox
def f(n: i32) -> i32:
  s = 0
  for i in range(len(n)):
    s = s + 1
  return s

//--- complex-len.ox
def f(x: buffer[complex_f32]) -> i32:
  s = 0
  for i in range(len(x)):
    s = s + 1
  return s

//--- loop-name.ox
def f(n: i32) -> i32:
  for = n
  return for

//--- index-mul.ox
def f(x: buffer[f32]) -> f32:
  s = 0.0
  for i in range(len(x)):
    s = s + x[i * 2]
  return s

//--- index-mix.ox
def f(x: buffer[f32], k: i32) -> f32:
  s = 0.0
  for i in range(len(x)):
    s = s + x[i + k]
  return s

//--- index-constant.ox
def f(x: buffer[f32]) -> f32:
  return x[2147483647 + 1]

//--- indent.ox
def f(n: i32) -> i32:
  s = 0
  for i in range(n):
      s = s + i
    s = s + 1
  return s

//--- empty-body.ox
def f(n: i32) -> i32:
  s = 0
  for i in range(n):
  s = s + 1
  return s

//--- fixed.ox
def f(x: tensor[q15,4]) -> tensor[q15,4]:
  for i in range(4):
    y = x
  return x

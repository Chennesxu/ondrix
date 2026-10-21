// RUN: split-file %s %t
// RUN: ondrix-compile %t/chain.ox | FileCheck %s --check-prefix=CHAIN
// RUN: ondrix-compile %t/noelse.ox | FileCheck %s --check-prefix=NOELSE
// RUN: ondrix-compile %t/while.ox | FileCheck %s --check-prefix=WHILE
// RUN: ondrix-compile %t/choose.ox | FileCheck %s --check-prefix=CHOOSE
// RUN: ondrix-compile %t/forget.ox | FileCheck %s --check-prefix=FORGET
// RUN: not ondrix-compile %t/condition.ox 2>&1 | FileCheck %s --check-prefix=CONDITION
// RUN: not ondrix-compile %t/whilecondition.ox 2>&1 | FileCheck %s --check-prefix=CONDITION
// RUN: not ondrix-compile %t/arms.ox 2>&1 | FileCheck %s --check-prefix=ARMS
// RUN: not ondrix-compile %t/return.ox 2>&1 | FileCheck %s --check-prefix=RETURN
// RUN: not ondrix-compile %t/break.ox 2>&1 | FileCheck %s --check-prefix=BREAK
// RUN: not ondrix-compile %t/stray.ox 2>&1 | FileCheck %s --check-prefix=STRAY
// RUN: not ondrix-compile %t/twoelse.ox 2>&1 | FileCheck %s --check-prefix=STRAY
// RUN: not ondrix-compile %t/scope.ox 2>&1 | FileCheck %s --check-prefix=SCOPE
// RUN: not ondrix-compile %t/carry.ox 2>&1 | FileCheck %s --check-prefix=CARRY
// RUN: not ondrix-compile %t/retype.ox 2>&1 | FileCheck %s --check-prefix=RETYPE
// RUN: not ondrix-compile %t/body.ox 2>&1 | FileCheck %s --check-prefix=BODY
// RUN: not ondrix-compile %t/fixed.ox 2>&1 | FileCheck %s --check-prefix=FIXED
// RUN: not ondrix-compile %t/else.ox 2>&1 | FileCheck %s --check-prefix=ELSE
// CHAIN: %[[V:.*]] = scf.if %arg1 -> (i32) {
// CHAIN: } else {
// CHAIN: %[[E:.*]] = scf.if %{{.*}} -> (i32) {
// CHAIN: arith.constant 3 : i32
// CHAIN: scf.yield %[[E]] : i32
// CHAIN: return %[[V]] : i32
// NOELSE: scf.if %arg1 {
// NOELSE-NEXT: arith.constant 1.0
// NOELSE-NOT: else
// NOELSE: return
// WHILE: %[[R:.*]] = scf.while (%[[I:.*]] = %{{.*}}) : (i32) -> i32 {
// WHILE-NEXT: %[[C:.*]] = arith.cmpi slt, %[[I]], %arg0 : i32
// WHILE-NEXT: scf.condition(%[[C]]) %[[I]] : i32
// WHILE: } do {
// WHILE: return %[[R]] : i32
// CHOOSE: %[[C:.*]] = scf.if %{{.*}} -> (i32) {
// CHOOSE-NEXT: scf.yield %arg1 : i32
// CHOOSE-NEXT: } else {
// CHOOSE-NEXT: arith.constant 0 : i32
// CHOOSE-NEXT: arith.subi
// CHOOSE: cf.assert {{.*}} "slice out of bounds"
// FORGET: cf.assert {{.*}} "slice out of bounds"
// CONDITION: error: a condition must be a bool scalar
// ARMS: error: the arms of a conditional expression must have one type
// RETURN: error: only the function body ends in return
// BREAK: error: break and continue are not supported
// STRAY: error: 'else' must follow an 'if' body at its column
// SCOPE: error: local 'v' is only visible inside the body that binds it
// CARRY: error: a loop can only carry scalar locals
// RETYPE: error: a reassignment must keep the local's type
// BODY: error: expected an indented body
// FIXED: error: loops and branches require an i32, i64, bool, f32, q15 or q31 function result
// ELSE: error: expected 'else' in the conditional expression

//--- chain.ox
def f(x: i32, c: bool) -> i32:
  v = x
  if c:
    v = 1
  elif x > 0:
    v = 2
  else:
    w = 3
    v = w
  return v

//--- noelse.ox
def f(y: mut buffer[f32], c: bool):
  if c:
    y[0] = 1.0

//--- while.ox
def f(n: i32) -> i32:
  i = 0
  while i < n:
    i = i + 2
  return i

//--- choose.ox
def f(x: buffer[f32], a: i32) -> f32:
  k = (a
       if a > 0 else -a)
  return dot(x[k:3], x[k:3], contract=off)

//--- forget.ox
def f(y: mut buffer[f32, 4], n: i32):
  k = 0
  while k < n:
    k = k + 1
  y[0] = dot(y[k:4], y[k:4], contract=off)

//--- condition.ox
def f(x: i32) -> i32:
  v = 0
  if x:
    v = 1
  return v

//--- whilecondition.ox
def f(x: f32) -> i32:
  i = 0
  while x:
    i = i + 1
  return i

//--- arms.ox
def f(x: i32, c: bool) -> i32:
  return x if c else 1.0

//--- return.ox
def f(x: i32) -> i32:
  v = 0
  if x > 0:
    return 1
  return v

//--- break.ox
def f(x: i32) -> i32:
  v = x
  while v > 0:
    break
  return v

//--- stray.ox
def f(x: i32) -> i32:
  v = 0
  else:
    v = 1
  return v

//--- twoelse.ox
def f(x: i32) -> i32:
  v = 0
  if x > 0:
    v = 1
  else:
    v = 2
  else:
    v = 3
  return v

//--- scope.ox
def f(x: i32) -> i32:
  if x > 0:
    v = 1
  else:
    v = 2
  return v

//--- carry.ox
def f(x: tensor[f32, 4], n: i32) -> tensor[f32, 4]:
  y = x
  i = 0
  while i < n:
    y = y * 2.0
    i = i + 1
  return y

//--- retype.ox
def f(x: i32, c: bool) -> i32:
  v = x
  if c:
    v = 1.0
  return v

//--- body.ox
def f(x: i32) -> i32:
  v = x
  if x > 0:
  return v

//--- fixed.ox
def f(x: tensor[complex_q15,4], c: bool) -> tensor[complex_q15,4]:
  if c:
    y = x
  return x

//--- else.ox
def f(x: i32, c: bool) -> i32:
  return x if c

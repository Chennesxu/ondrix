// RUN: split-file %s %t
// RUN: ondrix-compile %t/mean.ox | FileCheck %s --check-prefix=MEAN
// RUN: ondrix-compile %t/narrow.ox | FileCheck %s --check-prefix=NARROW
// RUN: ondrix-compile %t/round.ox | FileCheck %s --check-prefix=ROUND
// RUN: ondrix-compile %t/composed.ox | FileCheck %s --check-prefix=COMPOSED
// RUN: ondrix-compile %t/index.ox | ondrix-opt --canonicalize --convert-index-to-llvm="index-bitwidth=32" | FileCheck %s --check-prefix=INDEX
// RUN: not ondrix-compile %t/literal.ox 2>&1 | FileCheck %s --check-prefix=LITERAL
// RUN: not ondrix-compile %t/same.ox 2>&1 | FileCheck %s --check-prefix=SAME
// RUN: not ondrix-compile %t/fixed.ox 2>&1 | FileCheck %s --check-prefix=FIXED
// RUN: not ondrix-compile %t/bool.ox 2>&1 | FileCheck %s --check-prefix=BOOL
// RUN: not ondrix-compile %t/tensor.ox 2>&1 | FileCheck %s --check-prefix=TENSOR
// RUN: not ondrix-compile %t/findex.ox 2>&1 | FileCheck %s --check-prefix=FINDEX
// RUN: not ondrix-compile %t/rnarrow.ox 2>&1 | FileCheck %s --check-prefix=RNARROW
// RUN: not ondrix-compile %t/exact.ox 2>&1 | FileCheck %s --check-prefix=EXACT
// RUN: not ondrix-compile %t/tof32.ox 2>&1 | FileCheck %s --check-prefix=TOF32
// RUN: not ondrix-compile %t/toindex.ox 2>&1 | FileCheck %s --check-prefix=TOINDEX
// RUN: not ondrix-compile %t/fwrap.ox 2>&1 | FileCheck %s --check-prefix=FWRAP
// RUN: not ondrix-compile %t/mode.ox 2>&1 | FileCheck %s --check-prefix=MODE
// RUN: not ondrix-compile %t/constant.ox 2>&1 | FileCheck %s --check-prefix=CONSTANT
// RUN: not ondrix-compile %t/fold.ox 2>&1 | FileCheck %s --check-prefix=FOLD
// MEAN: %[[N:.*]] = index.casts %{{.*}} : index to i64
// MEAN-NEXT: %[[F:.*]] = arith.sitofp %[[N]] : i64 to f32
// MEAN-NEXT: arith.divf %{{.*}}, %[[F]] : f32
// NARROW: %[[HIGH:.*]] = arith.minsi %arg0, %c2147483647_i64
// NARROW-NEXT: %[[LOW:.*]] = arith.maxsi %[[HIGH]], %c-2147483648_i64
// NARROW-NEXT: arith.trunci %[[LOW]] : i64 to i32
// NARROW-NEXT: arith.trunci %arg1 : i64 to i32
// add() checks its operands again, after the first check filled the defaults.
// COMPOSED: %[[S:.*]] = arith.maxsi
// COMPOSED-NEXT: %[[A:.*]] = arith.trunci %[[S]] : i64 to i32
// COMPOSED-NEXT: %[[W:.*]] = arith.trunci %arg1 : i64 to i32
// COMPOSED-NEXT: arith.addi %[[A]], %[[W]] : i32
// ROUND: %[[ABOVE:.*]] = arith.cmpf oge, %arg0, %{{.*}} : f32
// ROUND: arith.cmpf uno, %arg0, %arg0
// ROUND: arith.fptosi %{{.*}} : f32 to i64
// ROUND: arith.select %[[ABOVE]], %c9223372036854775807_i64
// On a 32-bit index the round trip truncates and sign-extends, so the trap survives.
// INDEX: %[[T:.*]] = llvm.trunc %arg1 : i64 to i32
// INDEX: %[[S:.*]] = llvm.sext %[[T]] : i32 to i64
// INDEX-NEXT: %[[OK:.*]] = arith.cmpi eq, %[[S]], %arg1
// INDEX-NEXT: cf.assert %[[OK]], "conversion to index out of range"
// LITERAL: error: a convert operand needs a type; write the literal in the target type
// SAME: error: convert needs a target other than the operand's type
// FIXED: error: q15 and q31 convert with widen, narrow, quantize and dequantize
// BOOL: error: convert takes an i32, i64, f32 or index value to another of those types
// TENSOR: error: a tensor converts to i32, i64 or f32; index is a scalar type
// FINDEX: error: an f32 converts to i32 or i64, not to index
// RNARROW: error: an integer convert takes no rounding policy
// EXACT: error: this convert is exact and takes no rounding or overflow policy
// TOF32: error: a convert to f32 is one IEEE rounding and takes no rounding or overflow policy
// TOINDEX: error: a convert to index traps outside the target's index width and takes no policy
// FWRAP: error: an f32 convert saturates; wrapping an unbounded value is not a contract this operation offers
// MODE: error: unsupported rounding mode 'up'
// CONSTANT: error: constant index expression leaves the signed 32-bit range
// 2^32 wraps to the constant 0 (it would saturate to MAX), so the division is diagnosed.
// FOLD: error: integer division by zero

//--- mean.ox
def f(x: buffer[f32]) -> f32:
  return x[0] / convert(len(x), to=f32)

//--- narrow.ox
def f(v: i64, k: i64) -> i32:
  return convert(v, to=i32) + convert(k, to=i32, overflow=wrap)

//--- round.ox
def f(v: f32) -> i64:
  return convert(v, to=i64)

//--- composed.ox
def f(x: i64, y: i64) -> i32:
  return add(convert(x, to=i32), convert(y, to=i32, overflow=wrap))

//--- index.ox
def f(x: buffer[f32], k: i64) -> f32:
  return x[convert(k, to=index)]

//--- literal.ox
def f() -> f32:
  return convert(3, to=f32)

//--- same.ox
def f(k: i32) -> i32:
  return convert(k, to=i32)

//--- fixed.ox
def f(x: buffer[q15]) -> i32:
  return convert(x[0], to=i32)

//--- bool.ox
def f(b: bool) -> i32:
  return convert(b, to=i32)

//--- tensor.ox
def f(t: tensor[i64, 4]) -> tensor[i64, 4]:
  return t + convert(convert(t, to=index), to=i64)

//--- findex.ox
def f(x: buffer[f32], v: f32) -> f32:
  return x[convert(v, to=index)]

//--- rnarrow.ox
def f(w: i64) -> i32:
  return convert(w, to=i32, rounding=toward_zero)

//--- exact.ox
def f(k: i32) -> i64:
  return convert(k, to=i64, overflow=wrap)

//--- tof32.ox
def f(k: i32) -> f32:
  return convert(k, to=f32, rounding=toward_zero)

//--- toindex.ox
def f(x: buffer[f32], w: i64) -> f32:
  return x[convert(w, to=index, overflow=saturate)]

//--- fwrap.ox
def f(v: f32) -> i32:
  return convert(v, to=i32, overflow=wrap)

//--- mode.ox
def f(v: f32) -> i32:
  return convert(v, to=i32, rounding=up)

//--- constant.ox
def f(x: buffer[f32]) -> i64:
  n = 4294967296
  return convert(convert(n, to=index), to=i64)

//--- fold.ox
def f() -> i64:
  n = 4294967296
  return 7 / convert(convert(n, to=i32, overflow=wrap), to=i64)

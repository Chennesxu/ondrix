// RUN: split-file %s %t
// RUN: ondrix-compile %t/mac.ox | FileCheck %s --check-prefix=MAC
// RUN: ondrix-compile %t/literal.ox | FileCheck %s --check-prefix=LITERAL
// RUN: ondrix-compile %t/table.ox | FileCheck %s --check-prefix=TABLE
// RUN: ondrix-compile %t/compare.ox | FileCheck %s --check-prefix=COMPARE
// RUN: ondrix-compile %t/divide.ox | FileCheck %s --check-prefix=DIVIDE
// RUN: ondrix-compile %t/call.ox | FileCheck %s --check-prefix=CALL
// RUN: not ondrix-compile %t/mixed.ox 2>&1 | FileCheck %s --check-prefix=MIXED
// RUN: not ondrix-compile %t/broadcast.ox 2>&1 | FileCheck %s --check-prefix=BROADCAST
// RUN: not ondrix-compile %t/gain.ox 2>&1 | FileCheck %s --check-prefix=GAIN
// RUN: not ondrix-compile %t/spectrum.ox 2>&1 | FileCheck %s --check-prefix=SPECTRUM
// MAC: scf.for {{.*}} iter_args(%[[ACC:.*]] = %{{.*}}) -> (i32)
// MAC: %[[X:.*]] = ondrix.quantize {{.*}} : (i16) -> i32
// MAC: %[[H:.*]] = ondrix.quantize {{.*}} : (i16) -> i32
// MAC: %[[P:.*]] = ondrix.mult %[[X]], %[[H]] {{.*}} : (i32, i32) -> i32
// MAC: ondrix.add %[[ACC]], %[[P]] {{.*}} : (i32, i32) -> i32
// LITERAL: %[[C:.*]] = arith.constant -16384 : i16
// LITERAL: ondrix.sub %[[C]], %{{.*}} : (i16, i16) -> i16
// TABLE-COUNT-1: memref.global
// TABLE: ondrix.dot
// TABLE: ondsp.acc_export {{.*}} -> i16
// COMPARE: ondrix.negate {{.*}} : (i16) -> i16
// COMPARE: arith.cmpi sle
// MIXED: error: binary elementwise builtins require operands of the same element type and extent
// BROADCAST: error: binary elementwise builtins require operands of the same element type and extent
// GAIN: error: gain currently requires an input extent in [1, 4096]
// DIVIDE: ondrix.div %arg0 {divisor = 2 : i64
// CALL: arith.constant 2 : i16
// CALL: ondrix.mult %arg0, %{{.*}} {{.*}} : (i16, i16) -> i16
// CALL: arith.constant 3 : i16
// CALL: ondrix.mult %{{.*}}, %arg0 {{.*}} : (i16, i16) -> i16
// SPECTRUM: error: this builtin requires a tensor operand
//--- mac.ox
def mac(x: buffer[q15], h: buffer[q15]) -> q31:
  acc = 0
  for i in range(len(x)):
    acc = acc + widen(x[i], to=q31) * widen(h[i], to=q31)
  return acc

//--- literal.ox
def literal(x: q15) -> q15:
  return -16384 - x

//--- table.ox
def table(x: buffer[q15,4], c: constexpr[q15] = [1, 2, 3, 4]) -> q15:
  return dot(x, c) + dot(x, c)

//--- compare.ox
def compare(x: buffer[q15]) -> bool:
  return -x[1] <= 100

//--- mixed.ox
def mixed(x: buffer[q15]) -> q15:
  return x[0] + 0.5

//--- broadcast.ox
def broadcast(x: tensor[q15,4]) -> tensor[q15,4]:
  return x + x[0]

//--- gain.ox
def scaled(x: buffer[q15]) -> q15:
  return gain(x[0], gain=3)

//--- spectrum.ox
def spectrum(x: buffer[q15]) -> q15:
  return rfft(x[0])

//--- divide.ox
def halve(x: q15) -> q15:
  return x / 2

//--- call.ox
def doubled(x: q15) -> q31:
  return widen(mult(x, 2), to=q31) + widen(mult(3, x), to=q31)

// RUN: split-file %s %t
// RUN: ondrix-compile %t/stride.ox | ondrix-opt --canonicalize | FileCheck %s --check-prefix=STRIDE
// RUN: ondrix-compile %t/quotient.ox | FileCheck %s --check-prefix=QUOTIENT
// RUN: ondrix-compile %t/narrow.ox | ondrix-opt --canonicalize | FileCheck %s --check-prefix=NARROW
// RUN: not ondrix-compile %t/constant.ox 2>&1 | FileCheck %s --check-prefix=CONSTANT
// RUN: not ondrix-compile %t/zero.ox 2>&1 | FileCheck %s --check-prefix=ZERO
// STRIDE: %[[P:.*]] = index.mul %arg1, %[[TWO:.*]]
// STRIDE-NEXT: %[[Q:.*]] = index.divs %[[P]], %[[TWO]]
// STRIDE-NEXT: index.cmp eq(%[[Q]], %arg1)
// STRIDE-NEXT: cf.assert {{.*}}, "index arithmetic overflow"
// STRIDE: memref.load %arg0[%[[P]]]
// QUOTIENT: %[[D:.*]] = memref.dim %arg1
// QUOTIENT: index.cmp eq(%[[D]], %c-1)
// QUOTIENT: index.cmp ne(%[[D]], %{{.*}})
// QUOTIENT: cf.assert {{.*}}, "integer division by zero or overflow"
// QUOTIENT-NEXT: index.divs %{{.*}}, %[[D]]
// The product fits 64 bits but not 32, so a width-agreeing check keeps the trap.
// NARROW: index.divs
// NARROW-NEXT: index.cmp eq
// NARROW-NEXT: cf.assert {{.*}}, "index arithmetic overflow"
// CONSTANT: error: constant index expression leaves the signed 32-bit range
// ZERO: error: integer division by zero

//--- stride.ox
def f(x: buffer[f32]) -> f32:
  s = 0.0
  for i in range(len(x)):
    s = s + x[i * 2]
  return s

//--- quotient.ox
def f(x: buffer[f32], h: buffer[f32]) -> f32:
  return x[len(x) / len(h)]

//--- narrow.ox
def probe(x: tensor[f32,4]) -> f32:
  return x[len(x) * 1073741824 / 1073741824 - 1]

//--- constant.ox
def f(x: buffer[f32]) -> f32:
  return x[65536 * 32768]

//--- zero.ox
def f(x: buffer[f32]) -> f32:
  return x[len(x) / 0]

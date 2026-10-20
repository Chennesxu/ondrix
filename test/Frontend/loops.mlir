// RUN: split-file %s %t
// RUN: ondrix-compile %t/energy.ox | FileCheck %s --check-prefix=ENERGY
// RUN: ondrix-compile %t/tensor.ox | FileCheck %s --check-prefix=TENSOR
// RUN: ondrix-compile %t/reverse.ox | FileCheck %s --check-prefix=REVERSE
// RUN: ondrix-compile %t/offset.ox | FileCheck %s --check-prefix=OFFSET
// RUN: ondrix-compile %t/wide.ox | FileCheck %s --check-prefix=WIDE
// RUN: ondrix-compile %t/nested.ox | FileCheck %s --check-prefix=NESTED
// RUN: ondrix-compile %t/other.ox | FileCheck %s --check-prefix=CHECKED
// RUN: ondrix-compile %t/negative.ox | FileCheck %s --check-prefix=CHECKED
// RUN: ondrix-compile %t/static.ox | ondrix-opt --canonicalize | FileCheck %s --check-prefix=STATIC
// RUN: ondrix-compile %t/helper.ox | FileCheck %s --check-prefix=HELPER
// RUN: ondrix-compile %t/octal.ox | FileCheck %s --check-prefix=OCTAL
// ENERGY-LABEL: func.func @energy(%arg0: memref<?xf32>) -> f32
// ENERGY: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// ENERGY: %[[LEN:.*]] = memref.dim %arg0
// ENERGY: scf.for %[[I:.*]] = %{{.*}} to %[[LEN]] step %{{.*}} iter_args(%[[T:.*]] = %[[ZERO]]) -> (f32)
// ENERGY-NOT: cf.assert
// ENERGY: memref.load %arg0[%[[I]]]
// ENERGY: arith.mulf
// ENERGY: arith.addf %[[T]]
// TENSOR: scf.for {{.*}} -> (f32) {
// TENSOR-NOT: cf.assert
// TENSOR: tensor.extract
// REVERSE: arith.divui
// REVERSE: scf.while
// REVERSE: cf.assert {{.*}}, "index out of bounds"
// OFFSET: cf.assert {{.*}}, "index arithmetic overflow"
// OFFSET: cf.assert {{.*}}, "index out of bounds"
// WIDE: %[[POSITION:.*]] = arith.index_cast %arg1 : i64 to index
// WIDE: %[[LEN:.*]] = memref.dim
// WIDE: %[[WIDE:.*]] = arith.index_cast %[[LEN]] : index to i64
// WIDE: arith.cmpi ult, %arg1, %[[WIDE]] : i64
// WIDE: memref.load %arg0[%[[POSITION]]]
// NESTED: scf.for {{.*}} iter_args(%[[OUTER:.*]] = {{.*}}) -> (f32)
// CHECKED: cf.assert {{.*}}, "index out of bounds"
// NESTED: scf.for {{.*}} iter_args(%{{.*}} = %[[OUTER]]) -> (f32)
// Static len folds at 64 bits; only a width-agreeing index.cmp keeps the 32-bit trap.
// STATIC: index.cmp sge
// STATIC-NEXT: cf.assert {{.*}}, "index arithmetic overflow"
// HELPER: cf.assert {{.*}}, "index out of bounds"
// HELPER: tensor.extract %arg0
// HELPER: scf.for {{.*}} to %{{.*}} step
// HELPER: memref.load %arg1
// OCTAL: %[[STEP:.*]] = arith.constant 2 : i32
// OCTAL: arith.divui %{{.*}}, %[[STEP]]
//--- energy.ox
def energy(x: buffer[f32]) -> f32:
  total = 0.0
  for i in range(len(x)):
    total = total + x[i] * x[i]
  return total

//--- tensor.ox
def sum(x: tensor[f32,8]) -> f32:
  s = 0.0
  for i in range(2, len(x)):
    s = s + x[i]
  return s

//--- reverse.ox
def reverse(x: buffer[f32]) -> f32:
  s = 0.0
  for i in range(len(x) - 1, -1, -1):
    s = s + x[i]
  return s

//--- offset.ox
def offset(x: buffer[f32]) -> f32:
  s = 0.0
  for i in range(1, len(x)):
    s = s + x[i - 1]
  return s

//--- wide.ox
def wide(x: buffer[f32], k: i64) -> f32:
  return x[k]

//--- nested.ox
def nested(x: buffer[f32], y: buffer[f32]) -> f32:
  s = 0.0
  for i in range(len(x)):
    for j in range(len(y)):
      s = s + x[i] * y[j]
  return s

//--- other.ox
def other(x: buffer[f32], y: buffer[f32]) -> f32:
  s = 0.0
  for i in range(len(y)):
    s = s + x[i]
  return s

//--- negative.ox
def negative(x: buffer[f32]) -> f32:
  s = 0.0
  for i in range(-1, len(x)):
    s = s + x[i]
  return s

//--- static.ox
def probe(x: tensor[f32,4]) -> f32:
  return x[(len(x) + 2147483647) - 2147483647 - 1]

//--- helper.ox
def at(x: tensor[f32,4], k: i32) -> f32:
  return x[k]
def energy(x: buffer[f32]) -> f32:
  total = 0.0
  for i in range(len(x)):
    total = total + x[i] * x[i]
  return total
def f(x: tensor[f32,4], y: buffer[f32], k: i32) -> f32:
  return at(x, k) + energy(y)

//--- octal.ox
def steps(n: i32) -> i32:
  s = 0
  for i in range(0, n, 0o2):
    s = s + i
  return s

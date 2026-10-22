// RUN: split-file %s %t
// RUN: ondrix-compile %t/tail.ox | FileCheck %s --check-prefix=TAIL
// RUN: ondrix-compile %t/static.ox | FileCheck %s --check-prefix=STATIC
// RUN: ondrix-compile %t/array.ox | FileCheck %s --check-prefix=ARRAY
// RUN: ondrix-compile %t/readmut.ox --emit=c-header | FileCheck %s --check-prefix=READMUT
// RUN: ondrix-compile %t/callee.ox | FileCheck %s --check-prefix=CALLEE
// RUN: ondrix-compile %t/empty.ox | FileCheck %s --check-prefix=EMPTY
// RUN: not ondrix-compile %t/tensor.ox 2>&1 | FileCheck %s --check-prefix=TENSOR
// RUN: not ondrix-compile %t/passslice.ox 2>&1 | FileCheck %s --check-prefix=PASS
// RUN: not ondrix-compile %t/passarray.ox 2>&1 | FileCheck %s --check-prefix=PASS
// RUN: not ondrix-compile %t/step.ox 2>&1 | FileCheck %s --check-prefix=STEP
// RUN: not ondrix-compile %t/writeslice.ox 2>&1 | FileCheck %s --check-prefix=VIEW
// RUN: not ondrix-compile %t/writeview.ox 2>&1 | FileCheck %s --check-prefix=VIEW
// RUN: not ondrix-compile %t/ztype.ox 2>&1 | FileCheck %s --check-prefix=ZTYPE
// RUN: not ondrix-compile %t/zlength.ox 2>&1 | FileCheck %s --check-prefix=ZLENGTH
// RUN: not ondrix-compile %t/zconst.ox 2>&1 | FileCheck %s --check-prefix=ZCONST
// RUN: not ondrix-compile %t/rebind.ox 2>&1 | FileCheck %s --check-prefix=ONCE
// RUN: not ondrix-compile %t/scalarrebind.ox 2>&1 | FileCheck %s --check-prefix=ONCE
// RUN: not ondrix-compile %t/loopcarry.ox 2>&1 | FileCheck %s --check-prefix=ONCE
// RUN: not ondrix-compile %t/arith.ox 2>&1 | FileCheck %s --check-prefix=USE
// RUN: not ondrix-compile %t/value.ox 2>&1 | FileCheck %s --check-prefix=USE
// RUN: not ondrix-compile %t/range.ox 2>&1 | FileCheck %s --check-prefix=RANGE
// RUN: not ondrix-compile %t/order.ox 2>&1 | FileCheck %s --check-prefix=RANGE
// RUN: not ondrix-compile %t/negative.ox 2>&1 | FileCheck %s --check-prefix=RANGE
// RUN: not ondrix-compile %t/past.ox 2>&1 | FileCheck %s --check-prefix=RANGE
// RUN: not ondrix-compile %t/bound.ox 2>&1 | FileCheck %s --check-prefix=BOUND
// RUN: not ondrix-compile %t/base.ox 2>&1 | FileCheck %s --check-prefix=BASE
// RUN: not ondrix-compile %t/element.ox 2>&1 | FileCheck %s --check-prefix=ELEMENT
// TAIL: %[[START:.*]] = arith.index_cast %arg1 : i32 to index
// TAIL: cf.assert %{{.*}}, "slice out of bounds"
// TAIL: %[[SIZE:.*]] = arith.subi %{{.*}}, %[[START]] : index
// TAIL: memref.subview %arg0[%[[START]]] [%[[SIZE]]] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
// STATIC-NOT: cf.assert
// STATIC: %[[V:.*]] = memref.subview %arg0[%{{.*}}] [8] [1] : memref<16xf32> to memref<8xf32, strided<[1], offset: ?>>
// STATIC-NEXT: ondrix.dot %[[V]], %[[V]]
// ARRAY: %[[T:.*]] = memref.alloca() : memref<4xi16>
// ARRAY: memref.store %c0_i16, %[[T]][%{{.*}}] : memref<4xi16>
// ARRAY: %[[X:.*]] = memref.load %arg0
// ARRAY-NEXT: memref.store %[[X]], %[[T]][%{{.*}}] : memref<4xi16>
// ARRAY: %[[H:.*]] = memref.subview %arg1[%{{.*}}] [4] [1]
// ARRAY-NEXT: ondrix.dot %[[T]], %[[H]]
// READMUT: float ondrix_energy(const float *y, const float *h, uint64_t length);
// CALLEE: %[[W:.*]] = memref.subview %arg0
// CALLEE-NEXT: ondrix.dot %[[W]], %arg1
// EMPTY: %[[E:.*]] = memref.subview %arg0[%{{.*}}] [0] [1] : memref<4xf32> to memref<0xf32, strided<[1], offset: ?>>
// EMPTY: ondrix.dot %[[E]], %{{.*}} : (memref<0xf32
// TENSOR: error: scalar dot and fir currently require buffer operands
// PASS: error: an array or slice is not passed to a named function
// STEP: error: expected ']' after the slice; a slice has no step
// VIEW: error: a slice is a read-only view; write its elements
// ZTYPE: error: a local array holds f32, q15, q31, i32 or i64 elements
// ZLENGTH: error: a local array length must be in [1, 4096]
// ZCONST: error: expected the array length
// ONCE: error: an array or slice local is bound once
// USE: error: an array or slice is read through subscripts, len and slices, or as a dot, fir or rms operand
// RANGE: error: slice bounds are out of range
// BOUND: error: a slice bound must be an index, i32 or i64 scalar
// BASE: error: len, indexing and slices require a rank-one f32, q15, q31, i32 or i64 buffer, tensor, array or slice
// ELEMENT: error: a reduction inside an expression currently reads f32, q15 or q31 parameters, arrays or slices
//--- tail.ox
def tail(x: buffer[f32], a: i32) -> f32:
  w = x[a:]
  return w[0]

//--- static.ox
def level(x: buffer[f32, 16]) -> f32:
  return rms(x[2:10], contract=off)

//--- array.ox
def fill(x: buffer[q15], h: buffer[q15]) -> q15:
  t = zeros(q15, 4)
  for i in range(len(t)):
    t[i] = x[i]
  return dot(t, h[:4], accumulator=exact[40, saturate], rounding=nearest_ties_positive,
             overflow=saturate)

//--- readmut.ox
def energy(y: mut buffer[f32], h: buffer[f32]) -> f32:
  return dot(y, h, contract=off)

//--- callee.ox
def tail(x: buffer[f32], h: buffer[f32]) -> f32:
  return dot(x[1:], h, contract=off)

def f(x: buffer[f32], h: buffer[f32]) -> f32:
  return tail(x, h) + 1.0

//--- empty.ox
def f(x: buffer[f32, 4]) -> f32:
  return dot(x[4:4], x[0:0], contract=off)

//--- tensor.ox
def f(x: tensor[f32, 8], h: buffer[f32]) -> f32:
  w = x[2:6]
  return dot(w, h, contract=off)

//--- passslice.ox
def head(x: buffer[f32]) -> f32:
  return x[0]

def f(x: buffer[f32]) -> f32:
  return head(x[1:])

//--- passarray.ox
def head(x: buffer[f32]) -> f32:
  return x[0]

def f(x: buffer[f32]) -> f32:
  t = zeros(f32, 4)
  return head(t)

//--- step.ox
def f(x: buffer[f32]) -> f32:
  w = x[0:4:1]
  return w[0]

//--- writeslice.ox
def f(y: mut buffer[f32]):
  y[0:2] = 1.0

//--- writeview.ox
def f(y: mut buffer[f32]):
  w = y[0:2]
  w[0] = 1.0

//--- ztype.ox
def f(x: buffer[f32]) -> f32:
  t = zeros(bool, 4)
  return x[0]

//--- zlength.ox
def f(x: buffer[f32]) -> f32:
  t = zeros(f32, 5000)
  return t[0]

//--- zconst.ox
def f(x: buffer[f32]) -> f32:
  t = zeros(f32, len(x))
  return t[0]

//--- rebind.ox
def f(x: buffer[f32]) -> f32:
  t = zeros(f32, 4)
  t = zeros(f32, 4)
  return t[0]

//--- scalarrebind.ox
def f(x: buffer[f32]) -> f32:
  t = 1.0
  t = x[0:2]
  return t

//--- loopcarry.ox
def f(x: buffer[f32]) -> f32:
  w = x[0:1]
  for i in range(len(x)):
    w = x[i:]
  return w[0]

//--- arith.ox
def f(x: buffer[f32]) -> f32:
  t = zeros(f32, 4)
  return t + 1.0

//--- value.ox
def f(x: buffer[f32]) -> f32:
  s = x[0:2] + 1.0
  return s

//--- range.ox
def f(h: buffer[f32]) -> f32:
  t = zeros(f32, 4)
  return dot(t[2:6], h, contract=off)

//--- negative.ox
def f(x: buffer[f32], b: i32) -> f32:
  w = x[-1:b]
  return w[0]

//--- past.ox
def f(x: buffer[f32, 4], a: i32) -> f32:
  w = x[a:5]
  return w[0]

//--- order.ox
def f(x: buffer[f32]) -> f32:
  w = x[3:1]
  return w[0]

//--- bound.ox
def f(x: buffer[f32]) -> f32:
  w = x[1.0:2]
  return w[0]

//--- base.ox
def f(a: f32) -> f32:
  w = a[0:1]
  return w[0]

//--- element.ox
def f(x: buffer[f32], h: buffer[f32]) -> f32:
  return dot(x[0], h, contract=off)

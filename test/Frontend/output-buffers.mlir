// RUN: split-file %s %t
// RUN: ondrix-compile %t/scale.ox | FileCheck %s --check-prefix=SCALE
// RUN: ondrix-compile %t/scale.ox --emit=c-header | FileCheck %s --check-prefix=HEADER
// RUN: ondrix-compile %t/state.ox | FileCheck %s --check-prefix=STATE
// RUN: ondrix-compile %t/state.ox --emit=c-header | FileCheck %s --check-prefix=STATE-HEADER
// RUN: ondrix-compile %t/negate.ox --emit=c-header | FileCheck %s --check-prefix=NEGATE
// RUN: not ondrix-compile %t/tensor.ox 2>&1 | FileCheck %s --check-prefix=TENSOR
// RUN: not ondrix-compile %t/scalar.ox 2>&1 | FileCheck %s --check-prefix=SCALAR
// RUN: not ondrix-compile %t/readonly.ox 2>&1 | FileCheck %s --check-prefix=READONLY
// RUN: not ondrix-compile %t/local.ox 2>&1 | FileCheck %s --check-prefix=LOCAL
// RUN: not ondrix-compile %t/value.ox 2>&1 | FileCheck %s --check-prefix=VALUE
// RUN: not ondrix-compile %t/index.ox 2>&1 | FileCheck %s --check-prefix=INDEX
// RUN: not ondrix-compile %t/element.ox 2>&1 | FileCheck %s --check-prefix=ELEMENT
// RUN: not ondrix-compile %t/rank.ox 2>&1 | FileCheck %s --check-prefix=ELEMENT
// RUN: not ondrix-compile %t/kernel.ox 2>&1 | FileCheck %s --check-prefix=WHOLE
// RUN: not ondrix-compile %t/library.ox 2>&1 | FileCheck %s --check-prefix=WHOLE
// RUN: not ondrix-compile %t/alias.ox 2>&1 | FileCheck %s --check-prefix=WHOLE
// RUN: not ondrix-compile %t/callee.ox 2>&1 | FileCheck %s --check-prefix=CALLEE
// RUN: not ondrix-compile %t/nothing.ox 2>&1 | FileCheck %s --check-prefix=NOTHING
// RUN: not ondrix-compile %t/return.ox 2>&1 | FileCheck %s --check-prefix=RETURN
// RUN: not ondrix-compile %t/empty.ox 2>&1 | FileCheck %s --check-prefix=EMPTY
// SCALE: func.func @scale(%[[X:.*]]: memref<?xf32>, %[[Y:.*]]: memref<?xf32>, %[[G:.*]]: f32) attributes
// SCALE: scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step
// SCALE: cf.assert %{{.*}}, "index out of bounds"
// SCALE-NEXT: %[[V:.*]] = memref.load %[[X]][%[[I]]]
// SCALE-NEXT: %[[P:.*]] = arith.mulf %[[G]], %[[V]] : f32
// SCALE-NEXT: memref.store %[[P]], %[[Y]][%[[I]]] : memref<?xf32>
// SCALE: return{{$}}
// HEADER: void ondrix_scale(const float *x, uint64_t x_length, float *y, uint64_t y_length, float g);
// STATE: %[[OLD:.*]] = memref.load %arg0[%{{.*}}] : memref<?xi16>
// STATE: memref.store %[[OLD]], %arg0[%{{.*}}] : memref<?xi16>
// STATE: memref.store %arg1, %arg0[%{{.*}}] : memref<?xi16>
// STATE: return %{{.*}} : i16
// STATE-HEADER: int16_t ondrix_push(int16_t *d, uint64_t length, int16_t x);
// NEGATE: int16_t ondrix_readneg(const int16_t *x, uint64_t length);
// TENSOR: error: a tensor is a value; only a buffer can be mut
// SCALAR: error: expected 'buffer[...]' after 'mut'
// READONLY: error: parameter 'x' is read-only; only a mut buffer is written
// LOCAL: error: len and indexing currently require a rank-one f32, q15 or q31 buffer or tensor parameter
// VALUE: error: a write stores the buffer's element type
// INDEX: error: an index must be an index, i32 or i64 scalar
// ELEMENT: error: a mut buffer holds a rank-one f32, q15 or q31 sequence
// WHOLE: error: a mut buffer is only indexed, written or measured by len
// CALLEE: error: 'fill' writes a mut buffer; only the exported function may
// NOTHING: error: a function without a result must take a mut buffer
// RETURN: error: a function without a result has no return
// EMPTY: error: expected a statement
//--- scale.ox
def scale(x: buffer[f32], y: mut buffer[f32], g: f32):
  for i in range(len(y)):
    y[i] = g * x[i]

//--- state.ox
def push(d: mut buffer[q15], x: q15) -> q15:
  d[1] = d[0]
  d[0] = x
  return d[1]

//--- negate.ox
def readneg(x: mut buffer[q15]) -> q15:
  return negate(x[0])

//--- tensor.ox
def f(y: mut tensor[f32, 4]):
  y[0] = 1.0

//--- scalar.ox
def f(y: mut f32):
  y = 1.0

//--- readonly.ox
def f(x: buffer[f32], y: mut buffer[f32]):
  x[0] = y[0]

//--- local.ox
def f(y: mut buffer[f32]):
  t = y[0]
  t[0] = 1.0

//--- value.ox
def f(y: mut buffer[q15]):
  y[0] = 1.0

//--- index.ox
def f(y: mut buffer[f32]):
  y[1.0] = 1.0

//--- element.ox
def f(y: mut buffer[complex_q15, 4]):
  y[0] = y[1]

//--- rank.ox
def f(y: mut buffer[f32, 2, 2]):
  y[0] = 1.0

//--- kernel.ox
def f(y: mut buffer[f32], h: buffer[f32]) -> f32:
  return dot(y, h, contract=off)

//--- library.ox
def f(y: mut buffer[f32], h: buffer[f32]) -> f32:
  y[0] = 1.0
  return dot(y, h, contract=off) + 1.0

//--- alias.ox
def f(y: mut buffer[f32]) -> f32:
  t = y
  return t[0]

//--- callee.ox
def fill(y: mut buffer[f32]) -> f32:
  y[0] = 1.0
  return 1.0

def f(y: mut buffer[f32]) -> f32:
  return fill(y)

//--- nothing.ox
def f(x: buffer[f32]):
  t = x[0]
  t = t + 1.0

//--- return.ox
def f(y: mut buffer[f32]):
  y[0] = 1.0
  return y[0]

//--- empty.ox
def f(y: mut buffer[f32]):

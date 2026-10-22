// RUN: split-file %s %t
// RUN: ondrix-compile %t/buffer.ox | FileCheck %s --check-prefix=BUFFER
// RUN: ondrix-compile %t/divide.ox | FileCheck %s --check-prefix=DIVIDE
// RUN: ondrix-compile %t/convert.ox | FileCheck %s --check-prefix=CONVERT
// RUN: ondrix-compile %t/array.ox | FileCheck %s --check-prefix=ARRAY
// RUN: ondrix-compile %t/buffer.ox --emit=c-header | FileCheck %s --check-prefix=HEADER
// RUN: ondrix-compile %t/convert.ox --emit=c-header | FileCheck %s --check-prefix=TENSORHEADER
// RUN: not ondrix-compile %t/mixed.ox 2>&1 | FileCheck %s --check-prefix=MIXED
// RUN: not ondrix-compile %t/extent.ox 2>&1 | FileCheck %s --check-prefix=EXTENT
// RUN: not ondrix-compile %t/dynamic.ox 2>&1 | FileCheck %s --check-prefix=DYNAMIC
// RUN: not ondrix-compile %t/rank.ox 2>&1 | FileCheck %s --check-prefix=RANK
// RUN: not ondrix-compile %t/compare.ox 2>&1 | FileCheck %s --check-prefix=COMPARE
// RUN: not ondrix-compile %t/zero.ox 2>&1 | FileCheck %s --check-prefix=ZERO
// RUN: not ondrix-compile %t/store.ox 2>&1 | FileCheck %s --check-prefix=STORE
// RUN: not ondrix-compile %t/index.ox 2>&1 | FileCheck %s --check-prefix=INDEX
// RUN: not ondrix-compile %t/dot.ox 2>&1 | FileCheck %s --check-prefix=LIBRARY
// RUN: not ondrix-compile %t/second.ox 2>&1 | FileCheck %s --check-prefix=LIBRARY
// RUN: not ondrix-compile %t/array-dot.ox 2>&1 | FileCheck %s --check-prefix=LIBRARY
// RUN: not ondrix-compile %t/builtin.ox 2>&1 | FileCheck %s --check-prefix=BUILTIN
// BUFFER: func.func @f(%arg0: memref<?xi32>, %arg1: memref<?xi64>)
// BUFFER: %[[V:.*]] = memref.load %arg0[%{{.*}}] : memref<?xi32>
// BUFFER: %[[W:.*]] = arith.extsi %[[V]] : i32 to i64
// BUFFER: memref.store %{{.*}}, %arg1[%{{.*}}] : memref<?xi64>
// Each element's divisor is checked inside the loop that computes it.
// DIVIDE: scf.for
// DIVIDE: %[[D:.*]] = tensor.extract %arg1
// DIVIDE: arith.cmpi ne, %[[D]], %c0_i64
// DIVIDE: cf.assert %{{.*}}, "integer division by zero or overflow"
// DIVIDE-NEXT: arith.divsi %{{.*}}, %[[D]] : i64
// DIVIDE: arith.muli %{{.*}}, %c3_i64 : i64
// CONVERT: scf.for
// CONVERT: %[[E:.*]] = tensor.extract %arg0
// CONVERT: arith.cmpf uno, %[[E]], %[[E]] : f32
// CONVERT: arith.fptosi %{{.*}} : f32 to i32
// CONVERT: tensor.insert %{{.*}} : tensor<4xi32>
// ARRAY: memref.alloca() : memref<4xi64>
// ARRAY: memref.store %c0_i64
// HEADER: void ondrix_f(const int32_t *x, uint64_t x_length, int64_t *y, uint64_t y_length);
// TENSORHEADER: void ondrix_f(const float *x, int32_t *output);
// MIXED: error: arithmetic operands must have the same element type; no implicit conversion
// EXTENT: error: arithmetic tensor operands must have matching extents
// DYNAMIC: error: composable builtins currently require static operand extents
// RANK: error: i32 and i64 buffers and tensors are rank-one
// COMPARE: error: comparisons require scalar operands
// ZERO: error: integer division by zero
// STORE: error: a write stores the buffer's element type
// INDEX: error: a tensor converts to i32, i64 or f32; index is a scalar type
// LIBRARY: error: a reduction inside an expression currently reads f32, q15 or q31 parameters, arrays or slices
// BUILTIN: error: DSP builtins read f32, q15 or q31 sequences; i32 and i64 ones are read by indexing and arithmetic

//--- buffer.ox
def f(x: buffer[i32], y: mut buffer[i64]):
  for i in range(len(y)):
    y[i] = convert(x[i], to=i64) * 2

//--- divide.ox
def f(x: tensor[i64, 4], d: tensor[i64, 4]) -> tensor[i64, 4]:
  return x / d * 3

//--- convert.ox
def f(x: tensor[f32, 4]) -> tensor[i32, 4]:
  return convert(x, to=i32)

//--- array.ox
def f(x: buffer[i64]) -> i64:
  t = zeros(i64, 4)
  t[1] = x[0]
  return t[1] + t[0]

//--- mixed.ox
def f(x: tensor[i32, 4], y: tensor[i64, 4]) -> tensor[i64, 4]:
  return x + y

//--- extent.ox
def f(x: tensor[i32, 4], y: tensor[i32, 5]) -> tensor[i32, 4]:
  return x + y

//--- dynamic.ox
def f(x: tensor[i32]) -> tensor[i32]:
  return x + 1

//--- rank.ox
def f(x: buffer[i32, 2, 2]) -> i32:
  return 0

//--- compare.ox
def f(x: tensor[i32, 4]) -> bool:
  return x < x

//--- zero.ox
def f(x: tensor[i32, 4]) -> tensor[i32, 4]:
  return x / 0

//--- store.ox
def f(y: mut buffer[i32]):
  y[0] = 1.0

//--- index.ox
def f(t: tensor[i32, 4]) -> tensor[i64, 4]:
  return convert(convert(t, to=index), to=i64)

//--- dot.ox
def f(x: buffer[i32], h: buffer[f32]) -> f32:
  return dot(x, h, contract=off)

//--- second.ox
def f(x: buffer[f32], h: buffer[i32]) -> f32:
  return dot(x, h, contract=off) + 1.0

//--- array-dot.ox
def f(x: buffer[f32]) -> f32:
  t = zeros(i32, 4)
  return dot(x[0:4], t[0:4], contract=off)

//--- builtin.ox
def f(x: tensor[i32, 8]) -> tensor[i32, 8]:
  return moving_average(x, window=2)

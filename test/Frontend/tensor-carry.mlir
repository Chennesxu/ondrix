// RUN: split-file %s %t
// RUN: ondrix-compile %t/inplace.ox | FileCheck %s --check-prefix=INPLACE
// RUN: ondrix-compile %t/fixed.ox | FileCheck %s --check-prefix=FIXED
// RUN: ondrix-compile %t/alias.ox | FileCheck %s --check-prefix=ALIAS
// RUN: ondrix-compile %t/read.ox | FileCheck %s --check-prefix=READ
// RUN: not ondrix-compile %t/write.ox 2>&1 | FileCheck %s --check-prefix=WRITE
// RUN: not ondrix-compile %t/retype.ox 2>&1 | FileCheck %s --check-prefix=RETYPE
// RUN: not ondrix-compile %t/rebind.ox 2>&1 | FileCheck %s --check-prefix=REBIND
// RUN: not ondrix-compile %t/complex.ox 2>&1 | FileCheck %s --check-prefix=COMPLEX
// RUN: ondrix-compile %t/named.ox | FileCheck %s --check-prefix=NAMED
// The last elementwise loop writes into the carried value, so the loop yields its own iter_arg.
// INPLACE: scf.for %{{.*}} iter_args(%[[Y:.*]] = %arg0) -> (tensor<4xf32>)
// INPLACE: %[[T:.*]] = tensor.empty()
// INPLACE: scf.for {{.*}} iter_args(%{{.*}} = %[[T]])
// INPLACE: arith.mulf
// INPLACE: scf.for {{.*}} iter_args(%{{.*}} = %[[Y]])
// INPLACE: arith.addf
// Any other producer is copied into the carried value.
// FIXED: scf.for %{{.*}} iter_args(%[[Y:.*]] = %arg0) -> (tensor<4xi16>)
// FIXED: %[[S:.*]] = ondrix.add %[[Y]], %arg0
// FIXED: scf.for {{.*}} iter_args(%{{.*}} = %[[Y]])
// FIXED-NEXT: tensor.extract %[[S]]
// A binding that names carried state is a copy, taken before the state is rewritten.
// ALIAS: scf.for %{{.*}} iter_args(%[[Y:.*]] = %arg0, %[[S:.*]] = %arg0)
// ALIAS: %[[E:.*]] = tensor.empty()
// ALIAS: %[[Z:.*]] = scf.for {{.*}} iter_args(%{{.*}} = %[[E]])
// ALIAS-NEXT: tensor.extract %[[Y]]
// ALIAS: scf.for {{.*}} iter_args(%{{.*}} = %[[Y]])
// ALIAS: arith.mulf
// ALIAS: scf.for {{.*}} iter_args(%{{.*}} = %[[S]])
// ALIAS: tensor.extract %[[Z]]
// READ: %[[W:.*]] = tensor.extract_slice %{{.*}}[%{{.*}}] [2] [1] : tensor<4xf32> to tensor<2xf32>
// READ: tensor.dim %[[W]]
// READ: tensor.extract %[[W]]
// WRITE: error: a tensor is a value; write the elements of a zeros array
// RETYPE: error: a reassignment must keep the local's type
// REBIND: error: an array or slice local is bound once
// complex_f32 storage holds two slots per element, so a local is not measured either.
// COMPLEX: error: len, indexing and slices require a rank-one
// A value another name holds is copied in; its producer keeps its own destination.
// NAMED: %[[Z:.*]] = scf.for {{.*}} iter_args(%{{.*}} = %[[E:.*]]) -> (tensor<4xf32>)
// NAMED: scf.for %{{.*}} iter_args(%[[Y:.*]] = %arg0)
// NAMED: scf.for {{.*}} iter_args(%{{.*}} = %[[Y]])
// NAMED-NEXT: tensor.extract %[[Z]]

//--- inplace.ox
def f(x: tensor[f32, 4], n: i32) -> tensor[f32, 4]:
  y = x
  for i in range(n):
    y = y * 0.5 + x
  return y

//--- fixed.ox
def f(x: tensor[q15, 4], n: i32) -> tensor[q15, 4]:
  y = x
  for i in range(n):
    y = y + x
  return y

//--- alias.ox
def f(x: tensor[f32, 4], n: i32) -> tensor[f32, 4]:
  y = x
  s = x
  for i in range(n):
    z = y
    y = y * 2.0
    s = s + z
  return s

//--- read.ox
def f(x: tensor[f32, 4]) -> f32:
  y = x * 2.0
  w = y[1:3]
  s = 0.0
  for i in range(len(w)):
    s = s + w[i]
  return s

//--- write.ox
def f(x: tensor[f32, 4]) -> tensor[f32, 4]:
  y = x * 2.0
  y[0] = 1.0
  return y

//--- retype.ox
def f(x: tensor[f32, 4], n: i32) -> tensor[f32, 4]:
  y = x
  for i in range(n):
    y = convert(y, to=i32)
  return x

//--- rebind.ox
def f(x: tensor[f32, 4], n: i32) -> tensor[f32, 4]:
  y = x
  for i in range(n):
    y = y[0:2]
  return y

//--- complex.ox
def f(x: tensor[complex_f32, 4]) -> i32:
  y = x
  return convert(len(y), to=i32)

//--- named.ox
def f(x: tensor[f32, 4], n: i32) -> tensor[f32, 4]:
  z = x * 2.0
  y = x
  for i in range(n):
    y = z
  return y

// RUN: split-file %s %t
// RUN: ondrix-compile %t/loop.ox | FileCheck %s --check-prefix=LOOP
// RUN: ondrix-compile %t/repeat.ox | FileCheck %s --check-prefix=REPEAT
// RUN: ondrix-compile %t/callee.ox | FileCheck %s --check-prefix=CALLEE
// RUN: not ondrix-compile %t/local.ox 2>&1 | FileCheck %s --check-prefix=OPERAND
// RUN: not ondrix-compile %t/fixed.ox 2>&1 | FileCheck %s --check-prefix=OPERAND
// RUN: not ondrix-compile %t/tensor.ox 2>&1 | FileCheck %s --check-prefix=TENSOR
// RUN: not ondrix-compile %t/goertzel.ox 2>&1 | FileCheck %s --check-prefix=NESTED
// LOOP: scf.for {{.*}} iter_args(%[[S:.*]] = %{{.*}}) -> (f32)
// LOOP: %[[HALF:.*]] = arith.mulf %[[S]]
// LOOP: %[[DOT:.*]] = ondrix.dot %arg0, %arg1 {numeric = #ondsp.fp<format = f32, contract = off>}
// LOOP: arith.addf %[[HALF]], %[[DOT]]
// REPEAT: %[[DOT:.*]] = ondrix.dot %arg0, %arg0 {numeric = #ondsp.fp<format = f32, contract = fma>}
// REPEAT: arith.cmpf ogt, %[[DOT]], %arg1
// CALLEE: %[[FIR:.*]] = ondrix.fir %arg0, %arg1
// CALLEE: ondrix.dot %arg0, %arg0
// CALLEE: math.sqrt
// CALLEE: arith.subf %[[FIR]]
// OPERAND: error: a reduction inside an expression currently reads f32 parameters
// TENSOR: error: scalar dot and fir currently require buffer operands
// NESTED: error: nested calls are currently supported only by unary FFT-family builtins, a fir_filter input stage, and scalar dot/fir/rms
//--- loop.ox
def blend(x: buffer[f32], w: buffer[f32], n: i32) -> f32:
  s = 0.0
  for i in range(n):
    s = s * 0.5 + dot(x, w, contract=off)
  return s

//--- repeat.ox
def above(x: buffer[f32], t: f32) -> bool:
  return dot(x, x, contract=fma) > t

//--- callee.ox
def sample(x: buffer[f32,8], w: buffer[f32,8]) -> f32:
  return fir(x, w, contract=off)

def spread(x: buffer[f32,8], w: buffer[f32,8]) -> f32:
  return sample(x, w) - rms(x, contract=off)

//--- local.ox
def f(x: tensor[f32,4], w: buffer[f32]) -> f32:
  t = x * 2.0
  return dot(t, w, contract=off)

//--- fixed.ox
def f(q: buffer[q15]) -> f32:
  return dot(q, q) * 1.0

//--- tensor.ox
def f(x: tensor[f32,4]) -> f32:
  return dot(x, x, contract=off) + 1.0

//--- goertzel.ox
def f(x: tensor[f32,8]) -> f32:
  return goertzel(x, bin=1, contract=off) + 1.0

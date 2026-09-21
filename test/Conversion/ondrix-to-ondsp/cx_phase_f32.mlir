// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s
// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s --check-prefix=CONST

// The octant fold confines the ratio to [0, 1], so the whole construction is
// one division, one nine-term Horner chain, and exact power-of-two folds.
func.func @phase_off(%input: tensor<8xf32>) -> tensor<4xf32> {
  %out = ondrix.cx_phase %input {
    layout = #ondsp.cx_layout<interleaved>,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<8xf32>) -> tensor<4xf32>
  return %out : tensor<4xf32>
}

// CHECK-LABEL: func.func @phase_off
// CHECK: scf.for
// CHECK: %[[REAL:.*]] = tensor.extract
// CHECK: %[[IMAG:.*]] = tensor.extract
// CHECK: %[[A:.*]] = math.absf %[[REAL]]
// CHECK: %[[B:.*]] = math.absf %[[IMAG]]
// CHECK: %[[FOLD:.*]] = arith.cmpf ogt, %[[B]], %[[A]]
// CHECK: %[[HIGH:.*]] = arith.select %[[FOLD]], %[[B]], %[[A]]
// CHECK: %[[LOW:.*]] = arith.select %[[FOLD]], %[[A]], %[[B]]
// The origin has no ratio, so its divisor is substituted rather than its
// result selected.
// CHECK: %[[ORIGIN:.*]] = arith.cmpf oeq, %[[HIGH]], %{{.*}}
// CHECK: %[[DEN:.*]] = arith.select %[[ORIGIN]], %{{.*}}, %[[HIGH]]
// CHECK: %[[R:.*]] = arith.divf %[[LOW]], %[[DEN]]
// CHECK: %[[U:.*]] = arith.mulf %[[R]], %[[R]]
// CHECK-COUNT-8: arith.addf
// CHECK-NOT: math.fma
// CHECK: arith.select
// CHECK: tensor.insert

// The nine frozen coefficients reach the emitted module, the leading one
// being 1/(2*pi) rounded once to binary32.
// CONST-DAG: arith.constant 4.5357403E-4 : f32
// CONST-DAG: arith.constant -0.00255740178 : f32
// CONST-DAG: arith.constant 0.159154937 : f32

// The Horner chain is where the contract mode is spent: nine terms become
// eight fused updates instead of eight multiply-add pairs.
func.func @phase_fma(%input: tensor<8xf32>) -> tensor<4xf32> {
  %out = ondrix.cx_phase %input {
    layout = #ondsp.cx_layout<interleaved>,
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<8xf32>) -> tensor<4xf32>
  return %out : tensor<4xf32>
}

// CHECK-LABEL: func.func @phase_fma
// CHECK-COUNT-8: math.fma
// CHECK: arith.select

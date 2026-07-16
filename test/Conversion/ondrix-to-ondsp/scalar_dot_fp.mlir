// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s

func.func @dot_off(%lhs: f32, %rhs: f32) -> f32 {
  %result = ondrix.dot %lhs, %rhs {
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (f32, f32) -> f32
  return %result : f32
}

// CHECK-LABEL: func.func @dot_off
// CHECK: arith.mulf
// CHECK-NOT: math.fma

func.func @dot_fma(%lhs: f32, %rhs: f32) -> f32 {
  %result = ondrix.dot %lhs, %rhs {
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (f32, f32) -> f32
  return %result : f32
}

// CHECK-LABEL: func.func @dot_fma
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: math.fma {{.*}}, {{.*}}, %[[ZERO]] : f32

func.func @dot_fast(%lhs: f32, %rhs: f32) -> f32 {
  %result = ondrix.dot %lhs, %rhs {
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (f32, f32) -> f32
  return %result : f32
}

// CHECK-LABEL: func.func @dot_fast
// CHECK: math.fma {{.*}} fastmath<fast> : f32

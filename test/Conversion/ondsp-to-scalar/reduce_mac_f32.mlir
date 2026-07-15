// RUN: ondrix-opt %s --lower-ondsp-f32-reduce-to-scalar | FileCheck %s

func.func @reduce_mac_fma(%lhs: memref<8xf32>, %rhs: memref<8xf32>, %seed: f32) -> f32 {
  %keep = ondsp.assume_numeric %seed {numeric = #ondsp.fp<format = f32, contract = off>} : (f32) -> f32
  %0 = ondsp.reduce_mac %keep, %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = fma>} : (f32, memref<8xf32>, memref<8xf32>) -> f32
  return %0 : f32
}

// CHECK-LABEL: func.func @reduce_mac_fma
// CHECK: ondsp.assume_numeric
// CHECK: arith.constant 0 : index
// CHECK: arith.constant 8 : index
// CHECK: arith.constant 1 : index
// CHECK: scf.for {{.*}} iter_args({{.*}} = {{.*}}) -> (f32) {
// CHECK: memref.load
// CHECK: memref.load
// CHECK: math.fma
// CHECK: scf.yield
// CHECK-NOT: ondsp.reduce_mac

func.func @reduce_mac_off(%lhs: memref<8xf32>, %rhs: memref<8xf32>) -> f32 {
  %zero = arith.constant 0.0 : f32
  %0 = ondsp.reduce_mac %zero, %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = off>} : (f32, memref<8xf32>, memref<8xf32>) -> f32
  return %0 : f32
}

// CHECK-LABEL: func.func @reduce_mac_off
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK-NOT: math.fma
// CHECK-NOT: ondsp.reduce_mac

func.func @reduce_mac_fast(%lhs: memref<8xf32>, %rhs: memref<8xf32>) -> f32 {
  %zero = arith.constant 0.0 : f32
  %0 = ondsp.reduce_mac %zero, %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = fast>} : (f32, memref<8xf32>, memref<8xf32>) -> f32
  return %0 : f32
}

// CHECK-LABEL: func.func @reduce_mac_fast
// CHECK: math.fma {{.*}}fastmath<fast>{{.*}} : f32
// CHECK-NOT: ondsp.reduce_mac

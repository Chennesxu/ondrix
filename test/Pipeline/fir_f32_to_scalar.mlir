// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-to-scalar | FileCheck %s

func.func @fir_f32(%x: memref<8xf32>, %h: memref<8xf32>) -> f32 {
  %0 = ondrix.fir %x, %h {numeric = #ondsp.fp<format = f32, contract = fma>} : (memref<8xf32>, memref<8xf32>) -> f32
  return %0 : f32
}

// CHECK-LABEL: func.func @fir_f32
// CHECK: scf.for
// CHECK: memref.load
// CHECK: memref.load
// CHECK: math.fma
// CHECK-NOT: ondrix.fir
// CHECK-NOT: ondsp.reduce_mac

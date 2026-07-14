// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s

func.func @fir_f32(%x: memref<8xf32>, %h: memref<8xf32>) -> f32 {
  %0 = ondrix.fir %x, %h {numeric = #ondsp.fp<format = f32, contract = fma>, tag = "fir"} : (memref<8xf32>, memref<8xf32>) -> f32
  return %0 : f32
}

// CHECK-LABEL: func.func @fir_f32
// CHECK-NOT: tag =
// CHECK: ondsp.reduce_mac
// CHECK: #ondsp.fp<format = f32, contract = fma>
// CHECK-NOT: tag =

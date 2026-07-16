// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --vectorize-ondsp-q15-memref-reduce="vector-width=4" --normalize-ondsp-q15-vector-reduce --convert-ondsp-q15-to-scalar | FileCheck %s

func.func @fir_q15_vector(
    %input: memref<?xi16>, %coeffs: memref<?xi16>) -> i16 {
  %acc = ondrix.fir %input, %coeffs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (memref<?xi16>, memref<?xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %result : i16
}

// CHECK-LABEL: func.func @fir_q15_vector
// CHECK: cf.assert
// CHECK: scf.for
// CHECK: vector.load {{.*}} : memref<?xi16>, vector<4xi16>
// CHECK: vector.load {{.*}} : memref<?xi16>, vector<4xi16>
// CHECK: arith.extsi {{.*}} : vector<4xi16> to vector<4xi32>
// CHECK: arith.muli {{.*}} : vector<4xi32>
// CHECK: vector.extract
// CHECK: scf.for
// CHECK: memref.load
// CHECK: arith.muli {{.*}} : i32
// CHECK-NOT: ondrix.
// CHECK-NOT: ondsp.

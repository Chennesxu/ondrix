// RUN: ondrix-opt %s --convert-ondsp-q15-to-scalar | FileCheck %s

func.func @acc_add_product_q15(
    %acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    %product: i32) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  %result = ondsp.acc_add_product %acc, %product {
    product_numeric = #ondsp.fixed<signed, storage = i32, frac = 30>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i32) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @acc_add_product_q15(%{{.*}}: i40, %{{.*}}: i32) -> i40
// CHECK: arith.extsi {{.*}} : i40 to i41
// CHECK: arith.extsi {{.*}} : i32 to i41
// CHECK: arith.addi
// CHECK: arith.cmpi slt
// CHECK: arith.cmpi sgt
// CHECK: arith.select
// CHECK: arith.trunci {{.*}} : i41 to i40
// CHECK-NOT: ondsp.

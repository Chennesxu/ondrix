// RUN: ondrix-opt %s --convert-ondsp-fixed-to-scalar | FileCheck %s

func.func @acc_add_term_q15(
    %acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    %product: i32) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  %result = ondsp.acc_add_term %acc, %product {
    term_numeric = #ondsp.fixed<signed, storage = i32, frac = 30>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i32) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @acc_add_term_q15(%{{.*}}: i40, %{{.*}}: i32) -> i40
// CHECK: arith.extsi {{.*}} : i40 to i41
// CHECK: arith.extsi {{.*}} : i32 to i41
// CHECK: arith.addi
// CHECK: arith.cmpi slt
// CHECK: arith.cmpi sgt
// CHECK: arith.select
// CHECK: arith.trunci {{.*}} : i41 to i40
// CHECK-NOT: ondsp.

func.func @acc_add_wide_term_q15(
    %acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>,
    %term: i64) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap> {
  %result = ondsp.acc_add_term %acc, %term {
    term_numeric = #ondsp.fixed<signed, storage = i64, frac = 30>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, i64) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
}

// CHECK-LABEL: func.func @acc_add_wide_term_q15(%{{.*}}: i40, %{{.*}}: i64) -> i40
// CHECK: arith.extsi {{.*}} : i40 to i65
// CHECK: arith.extsi {{.*}} : i64 to i65
// CHECK: arith.addi
// CHECK: arith.trunci {{.*}} : i65 to i40
// CHECK-NOT: ondsp.

func.func @acc_add_inferred_width_q15(
    %acc: !ondsp.acc<storage = i48, frac = 30, signed, update_overflow = saturate>,
    %term: i64) -> !ondsp.acc<storage = i48, frac = 30, signed, update_overflow = saturate> {
  %result = ondsp.acc_add_term %acc, %term {
    term_numeric = #ondsp.fixed<signed, storage = i64, frac = 30>
  } : (!ondsp.acc<storage = i48, frac = 30, signed, update_overflow = saturate>, i64) -> !ondsp.acc<storage = i48, frac = 30, signed, update_overflow = saturate>
  return %result : !ondsp.acc<storage = i48, frac = 30, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @acc_add_inferred_width_q15(%{{.*}}: i48, %{{.*}}: i64) -> i48
// CHECK: arith.extsi {{.*}} : i48 to i65
// CHECK: arith.extsi {{.*}} : i64 to i65
// CHECK: arith.addi
// CHECK: arith.cmpi slt
// CHECK: arith.cmpi sgt
// CHECK: arith.trunci {{.*}} : i65 to i48
// CHECK-NOT: ondsp.

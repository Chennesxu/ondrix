// RUN: ondrix-opt %s --widen-ondsp-exact-accumulators | FileCheck %s

// 8 full Q15 products bound the web by 2^33 <= 2^34 - 1, so the declared
// wrap provably never fires and the web widens to the target domain.
// CHECK-LABEL: func.func @widens_static_loop(
// CHECK: ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate>
// CHECK: scf.for {{.*}} -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>)
// CHECK: ondsp.mac {{.*}} -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
// CHECK-NOT: update_overflow = wrap
func.func @widens_static_loop(%a: i16, %b: i16) -> i16 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %z = ondsp.acc_zero : !ondsp.acc<storage = i35, frac = 30, signed, update_overflow = wrap>
  %acc = scf.for %i = %c0 to %c8 step %c1 iter_args(%cur = %z) -> (!ondsp.acc<storage = i35, frac = 30, signed, update_overflow = wrap>) {
    %n = ondsp.mac %cur, %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i35, frac = 30, signed, update_overflow = wrap>, i16, i16) -> !ondsp.acc<storage = i35, frac = 30, signed, update_overflow = wrap>
    scf.yield %n : !ondsp.acc<storage = i35, frac = 30, signed, update_overflow = wrap>
  }
  %out = ondsp.acc_export %acc {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, rounding = #ondsp.rounding<nearest_ties_positive>, overflow = #ondsp.overflow<saturate>} : (!ondsp.acc<storage = i35, frac = 30, signed, update_overflow = wrap>) -> i16
  return %out : i16
}

// Three straight-line updates fit i33 (3 <= (2^32 - 1) >> 30).
// CHECK-LABEL: func.func @widens_straight_line(
// CHECK: ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate>
// CHECK-NOT: update_overflow = wrap
func.func @widens_straight_line(%a: i16, %b: i16) -> i32 {
  %z = ondsp.acc_zero : !ondsp.acc<storage = i33, frac = 30, signed, update_overflow = wrap>
  %s0 = ondsp.mac %z, %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i33, frac = 30, signed, update_overflow = wrap>, i16, i16) -> !ondsp.acc<storage = i33, frac = 30, signed, update_overflow = wrap>
  %s1 = ondsp.mac_sub %s0, %a, %a {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i33, frac = 30, signed, update_overflow = wrap>, i16, i16) -> !ondsp.acc<storage = i33, frac = 30, signed, update_overflow = wrap>
  %s2 = ondsp.mac %s1, %b, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i33, frac = 30, signed, update_overflow = wrap>, i16, i16) -> !ondsp.acc<storage = i33, frac = 30, signed, update_overflow = wrap>
  %out = ondsp.acc_export %s2 {dst = #ondsp.fixed<signed, storage = i32, frac = 15>, rounding = #ondsp.rounding<nearest_ties_positive>, overflow = #ondsp.overflow<saturate>} : (!ondsp.acc<storage = i33, frac = 30, signed, update_overflow = wrap>) -> i32
  return %out : i32
}

// 8 updates at i33 can reach 2^33 > 2^32 - 1: wrap is reachable, so the
// declaration keeps its meaning and the web stays untouched.
// CHECK-LABEL: func.func @keeps_reachable_wrap(
// CHECK: ondsp.acc_zero : <storage = i33, frac = 30, signed, update_overflow = wrap>
func.func @keeps_reachable_wrap(%a: i16, %b: i16) -> i16 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %z = ondsp.acc_zero : !ondsp.acc<storage = i33, frac = 30, signed, update_overflow = wrap>
  %acc = scf.for %i = %c0 to %c8 step %c1 iter_args(%cur = %z) -> (!ondsp.acc<storage = i33, frac = 30, signed, update_overflow = wrap>) {
    %n = ondsp.mac %cur, %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i33, frac = 30, signed, update_overflow = wrap>, i16, i16) -> !ondsp.acc<storage = i33, frac = 30, signed, update_overflow = wrap>
    scf.yield %n : !ondsp.acc<storage = i33, frac = 30, signed, update_overflow = wrap>
  }
  %out = ondsp.acc_export %acc {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, rounding = #ondsp.rounding<nearest_ties_positive>, overflow = #ondsp.overflow<saturate>} : (!ondsp.acc<storage = i33, frac = 30, signed, update_overflow = wrap>) -> i16
  return %out : i16
}

// A dynamic trip count admits no bound; the web stays untouched.
// CHECK-LABEL: func.func @keeps_dynamic_trip_count(
// CHECK: ondsp.acc_zero : <storage = i35, frac = 30, signed, update_overflow = wrap>
func.func @keeps_dynamic_trip_count(%a: i16, %b: i16, %n: index) -> i16 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %z = ondsp.acc_zero : !ondsp.acc<storage = i35, frac = 30, signed, update_overflow = wrap>
  %acc = scf.for %i = %c0 to %n step %c1 iter_args(%cur = %z) -> (!ondsp.acc<storage = i35, frac = 30, signed, update_overflow = wrap>) {
    %m = ondsp.mac %cur, %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i35, frac = 30, signed, update_overflow = wrap>, i16, i16) -> !ondsp.acc<storage = i35, frac = 30, signed, update_overflow = wrap>
    scf.yield %m : !ondsp.acc<storage = i35, frac = 30, signed, update_overflow = wrap>
  }
  %out = ondsp.acc_export %acc {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, rounding = #ondsp.rounding<nearest_ties_positive>, overflow = #ondsp.overflow<saturate>} : (!ondsp.acc<storage = i35, frac = 30, signed, update_overflow = wrap>) -> i16
  return %out : i16
}

// A second consumer of an intermediate breaks the single-flow shape.
// CHECK-LABEL: func.func @keeps_multi_use(
// CHECK: ondsp.acc_zero : <storage = i35, frac = 30, signed, update_overflow = wrap>
func.func @keeps_multi_use(%a: i16, %b: i16) -> (i16, i32) {
  %z = ondsp.acc_zero : !ondsp.acc<storage = i35, frac = 30, signed, update_overflow = wrap>
  %s0 = ondsp.mac %z, %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i35, frac = 30, signed, update_overflow = wrap>, i16, i16) -> !ondsp.acc<storage = i35, frac = 30, signed, update_overflow = wrap>
  %o1 = ondsp.acc_export %s0 {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, rounding = #ondsp.rounding<nearest_ties_positive>, overflow = #ondsp.overflow<saturate>} : (!ondsp.acc<storage = i35, frac = 30, signed, update_overflow = wrap>) -> i16
  %o2 = ondsp.acc_export %s0 {dst = #ondsp.fixed<signed, storage = i32, frac = 30>, rounding = #ondsp.rounding<toward_negative>, overflow = #ondsp.overflow<saturate>} : (!ondsp.acc<storage = i35, frac = 30, signed, update_overflow = wrap>) -> i32
  return %o1, %o2 : i16, i32
}

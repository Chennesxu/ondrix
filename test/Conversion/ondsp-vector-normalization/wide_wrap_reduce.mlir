// RUN: ondrix-opt %s --parallelize-ondsp-fixed-wrap-vector-reduce | FileCheck %s
// RUN: ondrix-opt %s --parallelize-ondsp-fixed-wrap-vector-reduce --normalize-ondsp-fixed-vector-reduce | FileCheck %s --check-prefix=ORDERED

// Q15 horizontal reassociation is admitted for signed frac30 accumulators
// wider than i40 only in wrap mode: exact-modulo reassociation is width
// independent up to the i64 carrier used by the horizontal sum. Saturating
// updates at the same width keep their ordered per-lane fold, because their
// legality needs the separate prefix-range proof.

func.func @parallel_wide_wrap(
    %initial: !ondsp.acc<storage = i64, frac = 30, signed, update_overflow = wrap>,
    %lhs: vector<8xi16>, %rhs: vector<8xi16>)
    -> !ondsp.acc<storage = i64, frac = 30, signed, update_overflow = wrap> {
  %result = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i64, frac = 30, signed, update_overflow = wrap>, vector<8xi16>, vector<8xi16>) -> !ondsp.acc<storage = i64, frac = 30, signed, update_overflow = wrap>
  return %result : !ondsp.acc<storage = i64, frac = 30, signed, update_overflow = wrap>
}

// CHECK-LABEL: func.func @parallel_wide_wrap
// CHECK: %[[PRODUCTS:.*]] = arith.muli {{.*}} : vector<8xi32>
// CHECK: %[[WIDE:.*]] = arith.extsi %[[PRODUCTS]] : vector<8xi32> to vector<8xi64>
// CHECK: %[[SUM:.*]] = vector.reduction <add>, %[[WIDE]] : vector<8xi64> into i64
// CHECK: ondsp.acc_add_term {{.*}}, %[[SUM]] {term_numeric = #ondsp.fixed<signed, storage = i64, frac = 30>}
// CHECK-NOT: vector.extract
// CHECK-NOT: ondsp.reduce_mac

func.func @preserve_wide_saturate(
    %initial: !ondsp.acc<storage = i64, frac = 30, signed, update_overflow = saturate>,
    %lhs: vector<8xi16>, %rhs: vector<8xi16>)
    -> !ondsp.acc<storage = i64, frac = 30, signed, update_overflow = saturate> {
  %result = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i64, frac = 30, signed, update_overflow = saturate>, vector<8xi16>, vector<8xi16>) -> !ondsp.acc<storage = i64, frac = 30, signed, update_overflow = saturate>
  return %result : !ondsp.acc<storage = i64, frac = 30, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @preserve_wide_saturate
// CHECK: ondsp.reduce_mac
// CHECK-NOT: vector.reduction

// The refused reduction normalizes to the ordered per-lane fold instead.
// ORDERED-LABEL: func.func @preserve_wide_saturate
// ORDERED: vector.extract %{{.*}}[0] : vector<8xi32>
// ORDERED-COUNT-8: ondsp.acc_add_term
// ORDERED-NOT: vector.reduction
// ORDERED-NOT: ondsp.reduce_mac

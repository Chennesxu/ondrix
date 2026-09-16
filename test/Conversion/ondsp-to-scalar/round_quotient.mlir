// RUN: ondrix-opt %s --convert-ondsp-fixed-to-scalar --split-input-file | FileCheck %s

// The divisor is tested once, before any division: a trapping policy asserts
// on it, and the division runs on a divisor the select has made positive,
// so no target-defined division by zero exists in either policy.
// CHECK-LABEL: func.func @trap
// CHECK: %[[D:.*]] = arith.extsi %arg1 : i16 to i31
// CHECK: %[[POS:.*]] = arith.cmpi sge, %[[D]], %[[ONE:.*]] : i31
// CHECK: cf.assert %[[POS]], "ondsp.round_quotient: the divisor is not positive"
// CHECK: %[[SAFE:.*]] = arith.select %[[POS]], %[[D]], %[[ONE]] : i31
// CHECK: arith.divsi %{{.*}}, %[[SAFE]] : i31
// CHECK: arith.remsi %{{.*}}, %[[SAFE]] : i31
// CHECK-NOT: arith.select %[[POS]]
func.func @trap(%x: i16, %y: i16) -> i16 {
  %0 = ondsp.round_quotient %x, %y {
    pre_shift_left = 15 : i64,
    rounding = #ondsp.rounding<nearest_ties_positive>,
    overflow = #ondsp.overflow<saturate>,
    nonpositive = #ondsp.nonpositive_divisor<trap>
  } : (i16, i16) -> i16
  return %0 : i16
}

// -----

// The saturating policy selects the dividend's signed rail, zero for a zero
// dividend, around the quotient; nothing asserts.
// CHECK-LABEL: func.func @saturate
// CHECK-NOT: cf.assert
// CHECK: %[[SCALED:.*]] = arith.shli
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i31
// CHECK: %[[POS:.*]] = arith.cmpi sge, %{{.*}}, %{{.*}} : i31
// CHECK: arith.divsi
// CHECK: %[[NEG:.*]] = arith.cmpi slt, %[[SCALED]], %[[ZERO]] : i31
// CHECK: %[[ISZERO:.*]] = arith.cmpi eq, %[[SCALED]], %[[ZERO]] : i31
// CHECK: %[[RAIL:.*]] = arith.select %[[NEG]], %{{.*}}, %{{.*}} : i16
// CHECK: %[[RAIL0:.*]] = arith.select %[[ISZERO]], %{{.*}}, %[[RAIL]] : i16
// CHECK: arith.select %[[POS]], %{{.*}}, %[[RAIL0]] : i16
func.func @saturate(%x: i16, %y: i16) -> i16 {
  %0 = ondsp.round_quotient %x, %y {
    pre_shift_left = 15 : i64,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<saturate>,
    nonpositive = #ondsp.nonpositive_divisor<saturate>
  } : (i16, i16) -> i16
  return %0 : i16
}

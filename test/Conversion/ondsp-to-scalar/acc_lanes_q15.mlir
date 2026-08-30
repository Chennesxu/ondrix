// RUN: ondrix-opt %s --convert-ondsp-fixed-to-scalar | FileCheck %s --implicit-check-not=ondsp.

// A multi-lane accumulator lowers to one storage element per lane and runs the
// identical per-lane arithmetic the single-lane path runs: the same i32 exact
// product, the same i41 update width, the same accumulator clamp bounds, and
// the same nearest-even export. The only difference is that the comparisons and
// selects are vector shaped. Compare the constants and widths below against
// mac_q15.mlir and acc_export_q15.mlir: they are the same numbers.

func.func @lane_lifecycle_saturate(%value: vector<8xi16>, %coefficient: i16) -> vector<8xi16> {
  %zero = ondsp.acc_zero
      : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>
  %accumulator = ondsp.mac %zero, %value, %coefficient {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>,
       vector<8xi16>, i16)
      -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>
  %result = ondsp.acc_export %accumulator {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>)
      -> vector<8xi16>
  return %result : vector<8xi16>
}

// CHECK-LABEL: func.func @lane_lifecycle_saturate(
// CHECK-SAME: %[[VALUE:.*]]: vector<8xi16>, %[[COEFFICIENT:.*]]: i16) -> vector<8xi16>
// CHECK: %[[ZERO:.*]] = arith.constant dense<0> : vector<8xi40>
// The declared per-lane broadcast of the scalar coefficient.
// CHECK: %[[SPLAT:.*]] = vector.broadcast %[[COEFFICIENT]] : i16 to vector<8xi16>
// CHECK: %[[VALUE_EXT:.*]] = arith.extsi %[[VALUE]] : vector<8xi16> to vector<8xi32>
// CHECK: %[[SPLAT_EXT:.*]] = arith.extsi %[[SPLAT]] : vector<8xi16> to vector<8xi32>
// CHECK: %[[PRODUCT:.*]] = arith.muli %[[VALUE_EXT]], %[[SPLAT_EXT]] : vector<8xi32>
// CHECK: %[[ACC_EXT:.*]] = arith.extsi %[[ZERO]] : vector<8xi40> to vector<8xi41>
// CHECK: %[[PRODUCT_EXT:.*]] = arith.extsi %[[PRODUCT]] : vector<8xi32> to vector<8xi41>
// CHECK: %[[UPDATED:.*]] = arith.addi %[[ACC_EXT]], %[[PRODUCT_EXT]] : vector<8xi41>
// The accumulator clamp bounds are the single-lane i40 bounds, splatted.
// CHECK: %[[MIN:.*]] = arith.constant dense<-549755813888> : vector<8xi41>
// CHECK: %[[MAX:.*]] = arith.constant dense<549755813887> : vector<8xi41>
// CHECK: %[[LOWER:.*]] = arith.maxsi %[[UPDATED]], %[[MIN]] : vector<8xi41>
// CHECK: %[[CLAMPED:.*]] = arith.minsi %[[LOWER]], %[[MAX]] : vector<8xi41>
// CHECK: %[[ACC:.*]] = arith.trunci %[[CLAMPED]] : vector<8xi41> to vector<8xi40>
// The export is the same nearest-even shift by 15 in quotient/remainder form.
// CHECK: %[[SHIFT:.*]] = arith.constant dense<15> : vector<8xi40>
// CHECK: %[[QUOTIENT:.*]] = arith.shrsi %[[ACC]], %[[SHIFT]] : vector<8xi40>
// CHECK: %[[LOW_BITS:.*]] = arith.trunci %[[ACC]] : vector<8xi40> to vector<8xi15>
// CHECK: %[[REMAINDER:.*]] = arith.extui %[[LOW_BITS]] : vector<8xi15> to vector<8xi40>
// CHECK: %[[EXPORT_ZERO:.*]] = arith.constant dense<0> : vector<8xi40>
// CHECK: %[[ONE:.*]] = arith.constant dense<1> : vector<8xi40>
// CHECK: %[[HALF:.*]] = arith.constant dense<16384> : vector<8xi40>
// CHECK: %[[ABOVE_HALF:.*]] = arith.cmpi ugt, %[[REMAINDER]], %[[HALF]] : vector<8xi40>
// CHECK: %[[EXACT_HALF:.*]] = arith.cmpi eq, %[[REMAINDER]], %[[HALF]] : vector<8xi40>
// CHECK: %[[LOW_BIT:.*]] = arith.andi %[[QUOTIENT]], %[[ONE]] : vector<8xi40>
// CHECK: %[[IS_ODD:.*]] = arith.cmpi ne, %[[LOW_BIT]], %[[EXPORT_ZERO]] : vector<8xi40>
// CHECK: %[[HALF_AND_ODD:.*]] = arith.andi %[[EXACT_HALF]], %[[IS_ODD]] : vector<8xi1>
// CHECK: %[[ROUND_UP:.*]] = arith.ori %[[ABOVE_HALF]], %[[HALF_AND_ODD]] : vector<8xi1>
// CHECK: %[[INCREMENT:.*]] = arith.select %[[ROUND_UP]], %[[ONE]], %[[EXPORT_ZERO]] : vector<8xi1>, vector<8xi40>
// CHECK: %[[ROUNDED:.*]] = arith.addi %[[QUOTIENT]], %[[INCREMENT]] : vector<8xi40>
// The destination clamp is the signed Q15 range, per lane.
// CHECK: %[[DST_MIN:.*]] = arith.constant dense<-32768> : vector<8xi40>
// CHECK: %[[DST_MAX:.*]] = arith.constant dense<32767> : vector<8xi40>
// CHECK: %[[DST_LOWER:.*]] = arith.maxsi %[[ROUNDED]], %[[DST_MIN]] : vector<8xi40>
// CHECK: %[[DST_CLAMPED:.*]] = arith.minsi %[[DST_LOWER]], %[[DST_MAX]] : vector<8xi40>
// CHECK: %[[RESULT:.*]] = arith.trunci %[[DST_CLAMPED]] : vector<8xi40> to vector<8xi16>
// CHECK: return %[[RESULT]] : vector<8xi16>

// The wrapping profile drops the accumulator clamp per lane exactly as the
// scalar one does.
func.func @lane_update_wrap(
    %accumulator: !ondsp.acc<storage = i34, frac = 30, signed, update_overflow = wrap, lanes = 4>,
    %value: vector<4xi16>, %coefficient: i16)
    -> !ondsp.acc<storage = i34, frac = 30, signed, update_overflow = wrap, lanes = 4> {
  %next = ondsp.mac %accumulator, %value, %coefficient {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i34, frac = 30, signed, update_overflow = wrap, lanes = 4>,
       vector<4xi16>, i16)
      -> !ondsp.acc<storage = i34, frac = 30, signed, update_overflow = wrap, lanes = 4>
  return %next
      : !ondsp.acc<storage = i34, frac = 30, signed, update_overflow = wrap, lanes = 4>
}

// CHECK-LABEL: func.func @lane_update_wrap(
// CHECK-SAME: %[[ACC:.*]]: vector<4xi34>, %[[VALUE:.*]]: vector<4xi16>, %[[COEFFICIENT:.*]]: i16) -> vector<4xi34>
// CHECK: vector.broadcast %[[COEFFICIENT]] : i16 to vector<4xi16>
// CHECK: arith.muli {{.*}} : vector<4xi32>
// CHECK: arith.extsi %[[ACC]] : vector<4xi34> to vector<4xi35>
// CHECK: %[[UPDATED:.*]] = arith.addi {{.*}} : vector<4xi35>
// CHECK-NOT: arith.cmpi
// CHECK: %[[RESULT:.*]] = arith.trunci %[[UPDATED]] : vector<4xi35> to vector<4xi34>
// CHECK: return %[[RESULT]] : vector<4xi34>

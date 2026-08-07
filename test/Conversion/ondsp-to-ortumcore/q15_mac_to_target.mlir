// RUN: ondrix-opt %s --convert-ondsp-to-ortumcore | FileCheck %s

// CHECK-LABEL: func.func @q15_mac(
// CHECK-SAME: %[[ACC:.*]]: !ortumcore.acc
// CHECK: %[[RESULT:.*]] = ortumcore.mac_add %[[ACC]], %{{.*}}, %{{.*}} : (!ortumcore.acc, i16, i16) -> !ortumcore.acc
// CHECK: return %[[RESULT]] : !ortumcore.acc
func.func @q15_mac(
    %acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    %a: i16, %b: i16)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  %0 = ondsp.mac %acc, %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %0 : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @q15_mac_sub(
// CHECK-SAME: %[[ACC:.*]]: !ortumcore.acc
// CHECK: %[[RESULT:.*]] = ortumcore.mac_sub %[[ACC]], %{{.*}}, %{{.*}} : (!ortumcore.acc, i16, i16) -> !ortumcore.acc
// CHECK: return %[[RESULT]] : !ortumcore.acc
func.func @q15_mac_sub(
    %acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    %a: i16, %b: i16)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  %0 = ondsp.mac_sub %acc, %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %0 : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @q15_export(
// CHECK-SAME: %[[ACC:.*]]: !ortumcore.acc
// CHECK: %[[OUT:.*]] = ortumcore.acc_out %[[ACC]] {shift = 15 : i64} : (!ortumcore.acc) -> i32
// CHECK-DAG: %[[MIN:.*]] = arith.constant -32768 : i32
// CHECK-DAG: %[[MAX:.*]] = arith.constant 32767 : i32
// CHECK: arith.cmpi slt, %[[OUT]], %[[MIN]]
// CHECK: arith.select
// CHECK: arith.cmpi sgt, %{{.*}}, %[[MAX]]
// CHECK: %[[CLAMPED:.*]] = arith.select
// CHECK: %[[NARROW:.*]] = arith.trunci %[[CLAMPED]] : i32 to i16
// CHECK: return %[[NARROW]] : i16
func.func @q15_export(
    %acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16 {
  %0 = ondsp.acc_export %acc {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, rounding = #ondsp.rounding<toward_negative>, overflow = #ondsp.overflow<saturate>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %0 : i16
}

// CHECK-LABEL: func.func @i32_export(
// CHECK-SAME: %[[ACC:.*]]: !ortumcore.acc
// CHECK: %[[OUT:.*]] = ortumcore.acc_out %[[ACC]] {shift = 0 : i64} : (!ortumcore.acc) -> i32
// CHECK-NOT: arith.cmpi
// CHECK: return %[[OUT]] : i32
func.func @i32_export(
    %acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i32 {
  %0 = ondsp.acc_export %acc {dst = #ondsp.fixed<signed, storage = i32, frac = 30>, rounding = #ondsp.rounding<toward_negative>, overflow = #ondsp.overflow<saturate>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i32
  return %0 : i32
}

// Discriminates the scf structural conversion: without it this whole-kernel
// loop shape fails to legalize at the scf.for carrying the accumulator.
// CHECK-LABEL: func.func @q15_loop_chain(
// CHECK: %[[ZERO:.*]] = ortumcore.acc_init : !ortumcore.acc
// CHECK: %[[LOOP:.*]] = scf.for %{{.*}} iter_args(%[[CUR:.*]] = %[[ZERO]]) -> (!ortumcore.acc)
// CHECK: %[[NEXT:.*]] = ortumcore.mac_add %[[CUR]], %{{.*}}, %{{.*}} : (!ortumcore.acc, i16, i16) -> !ortumcore.acc
// CHECK: scf.yield %[[NEXT]] : !ortumcore.acc
// CHECK: %[[OUT:.*]] = ortumcore.acc_out %[[LOOP]] {shift = 15 : i64} : (!ortumcore.acc) -> i32
func.func @q15_loop_chain(%lhs: i16, %rhs: i16, %count: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate>
  %acc = scf.for %i = %c0 to %count step %c1
      iter_args(%current = %zero) -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) {
    %next = ondsp.mac %current, %lhs, %rhs {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
    scf.yield %next : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  }
  %out = ondsp.acc_export %acc {dst = #ondsp.fixed<signed, storage = i32, frac = 15>, rounding = #ondsp.rounding<toward_negative>, overflow = #ondsp.overflow<saturate>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i32
  return %out : i32
}

// Ties-positive at shift 15 composes the narrower floor readout with one
// increment-and-halve at i64; no half enters the saturating accumulator.
// CHECK-LABEL: func.func @ntp_export(
// CHECK-SAME: %[[ACC:.*]]: !ortumcore.acc
// CHECK: %[[OUT:.*]] = ortumcore.acc_out %[[ACC]] {shift = 14 : i64} : (!ortumcore.acc) -> i32
// CHECK: %[[WIDE:.*]] = arith.extsi %[[OUT]] : i32 to i64
// CHECK: %[[ONE:.*]] = arith.constant 1 : i64
// CHECK: %[[INC:.*]] = arith.addi %[[WIDE]], %[[ONE]]
// CHECK: %[[HALF:.*]] = arith.shrsi %[[INC]], %[[ONE]]
// CHECK: %[[RES:.*]] = arith.trunci %[[HALF]] : i64 to i32
// CHECK: return %[[RES]] : i32
func.func @ntp_export(
    %acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i32 {
  %0 = ondsp.acc_export %acc {dst = #ondsp.fixed<signed, storage = i32, frac = 15>, rounding = #ondsp.rounding<nearest_ties_positive>, overflow = #ondsp.overflow<saturate>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i32
  return %0 : i32
}

// The i16 destination keeps the same composition and appends the clamp.
// CHECK-LABEL: func.func @ntp_export_i16(
// CHECK: ortumcore.acc_out %{{.*}} {shift = 14 : i64}
// CHECK: arith.shrsi
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.cmpi slt
// CHECK: arith.cmpi sgt
// CHECK: arith.trunci %{{.*}} : i32 to i16
func.func @ntp_export_i16(
    %acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16 {
  %0 = ondsp.acc_export %acc {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, rounding = #ondsp.rounding<nearest_ties_positive>, overflow = #ondsp.overflow<saturate>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %0 : i16
}

// Shift 0 is rounding-vacuous: the plain readout, no composition.
// CHECK-LABEL: func.func @ntp_export_raw(
// CHECK: %[[RAW:.*]] = ortumcore.acc_out %{{.*}} {shift = 0 : i64}
// CHECK-NOT: arith.addi
// CHECK: return %[[RAW]] : i32
func.func @ntp_export_raw(
    %acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i32 {
  %0 = ondsp.acc_export %acc {dst = #ondsp.fixed<signed, storage = i32, frac = 30>, rounding = #ondsp.rounding<nearest_ties_positive>, overflow = #ondsp.overflow<saturate>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i32
  return %0 : i32
}

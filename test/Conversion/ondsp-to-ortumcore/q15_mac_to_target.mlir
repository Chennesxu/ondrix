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

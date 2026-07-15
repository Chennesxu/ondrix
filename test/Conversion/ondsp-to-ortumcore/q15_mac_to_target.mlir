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

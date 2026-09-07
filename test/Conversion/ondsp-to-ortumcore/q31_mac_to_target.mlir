// RUN: ondrix-opt %s --convert-ondsp-to-ortumcore | FileCheck %s
// RUN: ondrix-opt %s --convert-ondsp-to-ortumcore --convert-ortumcore-to-ondsp-emulation | FileCheck %s --check-prefix=ROUNDTRIP

// The raw-high Q31 family lands on the Q15 accumulator; the export is the
// shift-0 saturating readout at the accumulator's own frac30 position.

// CHECK-LABEL: func.func @q31_raw_high_mac(
// CHECK-SAME: %[[ACC:.*]]: !ortumcore.acc
// CHECK: %[[ADD:.*]] = ortumcore.q31_mac_add %[[ACC]], %{{.*}}, %{{.*}} : (!ortumcore.acc, i32, i32) -> !ortumcore.acc
// CHECK: %[[SUB:.*]] = ortumcore.q31_mac_sub %[[ADD]], %{{.*}}, %{{.*}} : (!ortumcore.acc, i32, i32) -> !ortumcore.acc
// CHECK: ortumcore.acc_out %[[SUB]] {shift = 0 : i64} : (!ortumcore.acc) -> i32
// ROUNDTRIP-LABEL: func.func @q31_raw_high_mac(
// ROUNDTRIP: product = #ondsp.product<high_raw>
// ROUNDTRIP-NOT: ortumcore.
func.func @q31_raw_high_mac(
    %acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    %a: i32, %b: i32, %c: i32, %d: i32) -> i32 {
  %0 = ondsp.mac %acc, %a, %b {numeric = #ondsp.fixed<signed, storage = i32, frac = 31>, product = #ondsp.product<high_raw>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i32, i32) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %1 = ondsp.mac_sub %0, %c, %d {numeric = #ondsp.fixed<signed, storage = i32, frac = 31>, product = #ondsp.product<high_raw>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i32, i32) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %2 = ondsp.acc_export %1 {dst = #ondsp.fixed<signed, storage = i32, frac = 30>, rounding = #ondsp.rounding<toward_negative>, overflow = #ondsp.overflow<saturate>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i32
  return %2 : i32
}

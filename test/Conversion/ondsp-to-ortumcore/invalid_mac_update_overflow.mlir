// RUN: not ondrix-opt %s -split-input-file --convert-ondsp-to-ortumcore 2>&1 | FileCheck %s

func.func @q15_mac_requires_update_overflow(
    %acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, %a: i16, %b: i16)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  // CHECK: target MAC selection is disabled until accumulator update overflow semantics are explicit
  %0 = ondsp.mac %acc, %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %0 : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// -----

func.func @q15_mac_sub_requires_update_overflow(
    %acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, %a: i16, %b: i16)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  // CHECK: target MAC selection is disabled until accumulator update overflow semantics are explicit
  %0 = ondsp.mac_sub %acc, %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %0 : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// -----

func.func @q31_mac_requires_update_overflow(
    %acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, %a: i32, %b: i32)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  // CHECK: target MAC selection is disabled until accumulator update overflow semantics are explicit
  %0 = ondsp.mac %acc, %a, %b {numeric = #ondsp.fixed<signed, storage = i32, frac = 31>, product = #ondsp.product<high>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i32, i32) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %0 : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

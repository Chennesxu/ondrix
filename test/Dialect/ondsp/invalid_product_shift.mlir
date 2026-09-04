// RUN: ondrix-opt %s --split-input-file --verify-diagnostics

func.func @high_raw_with_shift(%acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, %a: i32, %b: i32) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  // expected-error @+1 {{the raw high product admits no requantization shift}}
  %r = ondsp.mac %acc, %a, %b {numeric = #ondsp.fixed<signed, storage = i32, frac = 31>, product = #ondsp.product<high_raw, shift = 1, rounding = nearest_even>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i32, i32) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %r : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// -----

func.func @accumulator_frac_mismatch(%acc: !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>, %a: i32, %b: i32) -> !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap> {
  // expected-error @+1 {{accumulator frac 62 does not match expected frac 59}}
  %r = ondsp.mac %acc, %a, %b {numeric = #ondsp.fixed<signed, storage = i32, frac = 31>, product = #ondsp.product<full, shift = 3, rounding = nearest_even>} : (!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>, i32, i32) -> !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>
  return %r : !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>
}

// RUN: ondrix-opt %s -split-input-file -verify-diagnostics

func.func @export_requires_matching_signedness(
    %acc: !ondsp.acc<storage = i40, frac = 30, signed>) -> i16 {
  // expected-error@+1 {{accumulator and destination signedness must match}}
  %0 = ondsp.acc_export %acc {dst = #ondsp.fixed<unsigned, storage = i16, frac = 15>, rounding = #ondsp.rounding<nearest_even>, overflow = #ondsp.overflow<saturate>} : (!ondsp.acc<storage = i40, frac = 30, signed>) -> i16
  return %0 : i16
}

// -----

func.func @export_rejects_fractional_upscale(
    %acc: !ondsp.acc<storage = i40, frac = 15, signed>) -> i32 {
  // expected-error@+1 {{destination frac must not exceed accumulator frac}}
  %0 = ondsp.acc_export %acc {dst = #ondsp.fixed<signed, storage = i32, frac = 30>, rounding = #ondsp.rounding<toward_zero>, overflow = #ondsp.overflow<wrap>} : (!ondsp.acc<storage = i40, frac = 15, signed>) -> i32
  return %0 : i32
}

// -----

func.func @export_requires_destination_storage(
    %acc: !ondsp.acc<storage = i40, frac = 30, signed>) -> i32 {
  // expected-error@+1 {{result type must match destination storage type}}
  %0 = ondsp.acc_export %acc {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, rounding = #ondsp.rounding<toward_negative>, overflow = #ondsp.overflow<saturate>} : (!ondsp.acc<storage = i40, frac = 30, signed>) -> i32
  return %0 : i32
}

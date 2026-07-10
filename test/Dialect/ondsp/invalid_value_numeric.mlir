// RUN: ondrix-opt %s -split-input-file -verify-diagnostics

func.func @assume_storage_mismatch(%x: i32) -> i32 {
  // expected-error@+1 {{input type does not match numeric storage type}}
  %0 = ondsp.assume_numeric %x {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>} : (i32) -> i32
  return %0 : i32
}

// -----

func.func @convert_destination_mismatch(%x: i16) -> i32 {
  // expected-error@+1 {{result type does not match numeric storage type}}
  %0 = ondsp.convert %x {src = #ondsp.fixed<signed, storage = i16, frac = 15>, dst = #ondsp.fixed<signed, storage = i16, frac = 7>} : (i16) -> i32
  return %0 : i32
}

// -----

func.func @reduce_fp_storage_mismatch(%a: f64, %b: f64) -> f32 {
  // expected-error@+1 {{lhs type does not match numeric storage type}}
  %0 = ondsp.reduce_mac %a, %b {numeric = #ondsp.fp<format = f32, contract = fma>} : (f64, f64) -> f32
  return %0 : f32
}

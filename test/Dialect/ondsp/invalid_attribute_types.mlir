// RUN: ondrix-opt %s -split-input-file -verify-diagnostics

func.func @sat_cast_rejects_invalid_numeric(%input: i32) -> i16 {
  // expected-error@+1 {{attribute 'numeric' failed to satisfy constraint: ondsp fixed-point or floating-point numeric attribute}}
  %0 = ondsp.sat_cast %input {numeric = "invalid"} : (i32) -> i16
  return %0 : i16
}

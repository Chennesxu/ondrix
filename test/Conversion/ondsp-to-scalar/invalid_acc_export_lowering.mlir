// RUN: not ondrix-opt %s --convert-ondsp-fixed-to-scalar --split-input-file 2>&1 | FileCheck %s

// The export destination domain is wider than Q15/Q30 but still closed: only
// a signed Q15 or signed i32 destination lowers from a signed frac30
// accumulator. Narrower storage and unsigned destinations fail closed until a
// consumer defines their range and sign semantics.

// CHECK: error: 'ondsp.acc_export' op fixed scalar lowering supports a signed frac30 accumulator of at least 32 bits to a signed Q15 or signed i32 destination, and i64/frac62 to Q31 export

func.func @export_narrow_destination(
    %acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) -> i8 {
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i8, frac = 7>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) -> i8
  return %result : i8
}

// -----

// CHECK: error: 'ondsp.acc_export' op fixed scalar lowering supports a signed frac30 accumulator of at least 32 bits to a signed Q15 or signed i32 destination, and i64/frac62 to Q31 export

func.func @export_wide_destination(
    %acc: !ondsp.acc<storage = i64, frac = 30, signed, update_overflow = wrap>) -> i64 {
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i64, frac = 24>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (!ondsp.acc<storage = i64, frac = 30, signed, update_overflow = wrap>) -> i64
  return %result : i64
}

// RUN: ondrix-opt %s -split-input-file -verify-diagnostics

func.func @fixed_storage_must_be_signless(%input: si16) -> si16 {
  // expected-error@+1 {{fixed numeric storage must use a signless integer type}}
  %0 = ondsp.assume_numeric %input {numeric = #ondsp.fixed<signed, storage = si16, frac = 15>} : (si16) -> si16
  return %0 : si16
}

// -----

// expected-error@+2 {{accumulator storage must use a signless integer type}}
func.func @accumulator_storage_must_be_signless(%input: i16)
    -> !ondsp.acc<storage = ui40, frac = 30, unsigned, update_overflow = saturate> {
  %0 = ondsp.acc_init %input : (i16) -> !ondsp.acc<storage = ui40, frac = 30, unsigned, update_overflow = saturate>
  return %0 : !ondsp.acc<storage = ui40, frac = 30, unsigned, update_overflow = saturate>
}

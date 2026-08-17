// RUN: ondrix-opt %s --split-input-file --verify-diagnostics

func.func @zero_width_rejected(%cursor: index, %step: index) -> index {
  // expected-error@+1 {{requires a reversal width between 1 and 32}}
  %next = ondsp.bitrev_add %cursor, %step {width = 0 : i64} : (index, index) -> index
  return %next : index
}

// -----

// The 32-bit target progression is the widest provable composition.
func.func @wide_width_rejected(%cursor: index, %step: index) -> index {
  // expected-error@+1 {{requires a reversal width between 1 and 32}}
  %next = ondsp.bitrev_add %cursor, %step {width = 33 : i64} : (index, index) -> index
  return %next : index
}

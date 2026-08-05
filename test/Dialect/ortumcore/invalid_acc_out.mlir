// RUN: ondrix-opt %s -split-input-file -verify-diagnostics

func.func @readout_shift_above_range(%acc: !ortumcore.acc) -> i32 {
  // expected-error @+1 {{accumulator readout shift must lie in [0, 15]}}
  %0 = ortumcore.acc_out %acc {shift = 16 : i64} : (!ortumcore.acc) -> i32
  return %0 : i32
}

// -----

func.func @readout_shift_below_range(%acc: !ortumcore.acc) -> i32 {
  // expected-error @+1 {{accumulator readout shift must lie in [0, 15]}}
  %0 = ortumcore.acc_out %acc {shift = -1 : i64} : (!ortumcore.acc) -> i32
  return %0 : i32
}

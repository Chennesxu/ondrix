// RUN: ondrix-opt %s --unroll-ondsp-fixed-mac-loops | FileCheck %s --check-prefix=OFF
// RUN: ondrix-opt %s --unroll-ondsp-fixed-mac-loops=max-carried-window=8 | FileCheck %s --check-prefix=TIGHT

// The term budget cannot express this choice: a 64-term dot carries nothing
// across a loop while a 12-tap filter carries eleven values, so one budget
// that unrolls the dot must also unroll the filter.

// A sliding window of twelve advancing by one leaves eleven values live in
// registers once the taps are straight-lined, which is what declines it.
// OFF-LABEL: func.func @slides_twelve
// OFF-NOT: iter_args
// TIGHT-LABEL: func.func @slides_twelve
// TIGHT: scf.for {{.*}} iter_args
func.func @slides_twelve(%signal: memref<64xi16>, %taps: memref<12xi16>, %out: memref<32xi16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c12 = arith.constant 12 : index
  %c32 = arith.constant 32 : index
  scf.for %n = %c0 to %c32 step %c1 {
    %seed = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
    %sum = scf.for %k = %c0 to %c12 step %c1 iter_args(%acc = %seed) -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) {
      %index = arith.addi %n, %k : index
      %a = memref.load %signal[%index] : memref<64xi16>
      %b = memref.load %taps[%k] : memref<12xi16>
      %next = ondsp.mac %acc, %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
      scf.yield %next : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
    }
    %value = ondsp.acc_export %sum {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, overflow = #ondsp.overflow<wrap>, rounding = #ondsp.rounding<nearest_even>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) -> i16
    memref.store %value, %out[%n] : memref<32xi16>
  }
  return
}

// -----

// The same twelve taps decimating by sixteen share nothing between trips, so
// the window is not what declines a shape -- the OVERLAP is.
// OFF-LABEL: func.func @decimates_past_its_window
// TIGHT-LABEL: func.func @decimates_past_its_window
// TIGHT-NOT: iter_args
func.func @decimates_past_its_window(%signal: memref<512xi16>, %taps: memref<12xi16>, %out: memref<32xi16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c12 = arith.constant 12 : index
  %c16 = arith.constant 16 : index
  %c512 = arith.constant 512 : index
  scf.for %n = %c0 to %c512 step %c16 {
    %seed = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
    %sum = scf.for %k = %c0 to %c12 step %c1 iter_args(%acc = %seed) -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) {
      %index = arith.addi %n, %k : index
      %a = memref.load %signal[%index] : memref<512xi16>
      %b = memref.load %taps[%k] : memref<12xi16>
      %next = ondsp.mac %acc, %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
      scf.yield %next : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
    }
    %value = ondsp.acc_export %sum {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, overflow = #ondsp.overflow<wrap>, rounding = #ondsp.rounding<nearest_even>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) -> i16
    memref.store %value, %out[%c0] : memref<32xi16>
  }
  return
}

// -----

// A reduction with no enclosing loop carries nothing however long it is, so
// the same limit that declined twelve sliding taps leaves sixty-four alone.
// TIGHT-LABEL: func.func @dot_carries_nothing
// TIGHT-NOT: iter_args
func.func @dot_carries_nothing(%lhs: memref<64xi16>, %rhs: memref<64xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %seed = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %sum = scf.for %k = %c0 to %c64 step %c1 iter_args(%acc = %seed) -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) {
    %a = memref.load %lhs[%k] : memref<64xi16>
    %b = memref.load %rhs[%k] : memref<64xi16>
    %next = ondsp.mac %acc, %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
    scf.yield %next : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  }
  return %sum : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
}

// -----

// The same window before bufferization: the reduction still reads
// `base[outer + inner]`, only through a tensor, and the limit must see it
// there too because that is the form the target route prices.
// TIGHT-LABEL: func.func @slides_twelve_on_tensors
// TIGHT: scf.for {{.*}} iter_args
// TIGHT: scf.for {{.*}} iter_args
func.func @slides_twelve_on_tensors(%signal: tensor<64xi16>, %taps: tensor<12xi16>, %out: tensor<32xi16>) -> tensor<32xi16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c12 = arith.constant 12 : index
  %c32 = arith.constant 32 : index
  %result = scf.for %n = %c0 to %c32 step %c1 iter_args(%dst = %out) -> (tensor<32xi16>) {
    %seed = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
    %sum = scf.for %k = %c0 to %c12 step %c1 iter_args(%acc = %seed) -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) {
      %index = arith.addi %n, %k : index
      %a = tensor.extract %signal[%index] : tensor<64xi16>
      %b = tensor.extract %taps[%k] : tensor<12xi16>
      %next = ondsp.mac %acc, %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
      scf.yield %next : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
    }
    %value = ondsp.acc_export %sum {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, overflow = #ondsp.overflow<wrap>, rounding = #ondsp.rounding<nearest_even>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) -> i16
    %stored = tensor.insert %value into %dst[%n] : tensor<32xi16>
    scf.yield %stored : tensor<32xi16>
  }
  return %result : tensor<32xi16>
}

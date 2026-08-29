// RUN: ondrix-opt %s --unroll-ondsp-fixed-mac-loops --split-input-file | FileCheck %s

// The tap-loop shape the tensor-form FIR lowering emits: the bound arrives
// as tensor.dim of a static tensor and counts as a compile-time constant.
// CHECK-LABEL: func.func @unrolls_static_tap_loop(
// CHECK-NOT: scf.for
// CHECK-COUNT-5: ondsp.mac
// CHECK-NOT: scf.for
// CHECK: ondsp.acc_export
func.func @unrolls_static_tap_loop(%input: tensor<64xi16>, %coefficients: tensor<5xi16>, %offset: index) -> i16 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %taps = tensor.dim %coefficients, %c0 : tensor<5xi16>
  %z = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %acc = scf.for %tap = %c0 to %taps step %c1 iter_args(%chain = %z) -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) {
    %position = arith.addi %offset, %tap : index
    %sample = tensor.extract %input[%position] : tensor<64xi16>
    %coefficient = tensor.extract %coefficients[%tap] : tensor<5xi16>
    %next = ondsp.mac %chain, %sample, %coefficient {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
    scf.yield %next : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  }
  %out = ondsp.acc_export %acc {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, rounding = #ondsp.rounding<nearest_ties_positive>, overflow = #ondsp.overflow<saturate>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %out : i16
}

// -----

// A dynamic bound has no compile-time trip count to expand.
// CHECK-LABEL: func.func @keeps_dynamic_bound(
// CHECK: scf.for
// CHECK: ondsp.mac
func.func @keeps_dynamic_bound(%samples: memref<64xi16>, %coefficients: memref<64xi16>, %count: index) -> i16 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %z = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %acc = scf.for %i = %c0 to %count step %c1 iter_args(%chain = %z) -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) {
    %sample = memref.load %samples[%i] : memref<64xi16>
    %coefficient = memref.load %coefficients[%i] : memref<64xi16>
    %next = ondsp.mac %chain, %sample, %coefficient {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
    scf.yield %next : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  }
  %out = ondsp.acc_export %acc {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, rounding = #ondsp.rounding<nearest_ties_positive>, overflow = #ondsp.overflow<saturate>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %out : i16
}

// -----

// One term past the measured straight-line bound keeps the loop form.
// CHECK-LABEL: func.func @keeps_trip_past_bound(
// CHECK: scf.for
// CHECK: ondsp.mac
func.func @keeps_trip_past_bound(%samples: memref<65xi16>, %coefficients: memref<65xi16>) -> i16 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c65 = arith.constant 65 : index
  %z = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %acc = scf.for %i = %c0 to %c65 step %c1 iter_args(%chain = %z) -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) {
    %sample = memref.load %samples[%i] : memref<65xi16>
    %coefficient = memref.load %coefficients[%i] : memref<65xi16>
    %next = ondsp.mac %chain, %sample, %coefficient {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
    scf.yield %next : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  }
  %out = ondsp.acc_export %acc {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, rounding = #ondsp.rounding<nearest_ties_positive>, overflow = #ondsp.overflow<saturate>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %out : i16
}

// -----

// A multi-lane accumulator is not this pass's chain; only acc_zero, mac, and
// acc_export accept lanes > 1 and their loop form stays for its own consumer.
// CHECK-LABEL: func.func @keeps_multi_lane_chain(
// CHECK: scf.for
// CHECK: ondsp.mac
func.func @keeps_multi_lane_chain(%pairs: memref<8xvector<2xi16>>, %coefficients: memref<8xi16>) -> vector<2xi16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %z = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 2>
  %acc = scf.for %i = %c0 to %c8 step %c1 iter_args(%chain = %z) -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 2>) {
    %pair = memref.load %pairs[%i] : memref<8xvector<2xi16>>
    %coefficient = memref.load %coefficients[%i] : memref<8xi16>
    %next = ondsp.mac %chain, %pair, %coefficient {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 2>, vector<2xi16>, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 2>
    scf.yield %next : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 2>
  }
  %out = ondsp.acc_export %acc {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, rounding = #ondsp.rounding<nearest_ties_positive>, overflow = #ondsp.overflow<saturate>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 2>) -> vector<2xi16>
  return %out : vector<2xi16>
}

// -----

// An accumulator that only rides the loop without a mac is outside the
// measured motive; the loop keeps its form.
// CHECK-LABEL: func.func @keeps_mac_free_loop(
// CHECK: scf.for
func.func @keeps_mac_free_loop(%unused: memref<4xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %z = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %acc = scf.for %i = %c0 to %c4 step %c1 iter_args(%chain = %z) -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) {
    scf.yield %chain : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  }
  return %acc : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

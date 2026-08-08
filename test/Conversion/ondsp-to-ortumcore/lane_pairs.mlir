// RUN: ondrix-opt %s --convert-ondsp-lane-pairs-to-ortumcore --split-input-file | FileCheck %s

!pair = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 2>

// A pair web becomes two target accumulators advanced by one dmac per mac,
// with the broadcast coefficient in both multiplier halves and one readout per lane.
func.func @straight_line(%values: vector<2xi16>, %coefficient: i16) -> vector<2xi32> {
  %zero = ondsp.acc_zero : !pair
  %acc = ondsp.mac %zero, %values, %coefficient {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!pair, vector<2xi16>, i16) -> !pair
  %out = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i32, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<saturate>
  } : (!pair) -> vector<2xi32>
  return %out : vector<2xi32>
}

// CHECK-LABEL: func.func @straight_line(
// CHECK-SAME: %[[VALUES:.*]]: vector<2xi16>, %[[COEFFICIENT:.*]]: i16) -> vector<2xi32>
// CHECK-DAG: %[[INIT0:.*]] = ortumcore.acc_init : !ortumcore.acc
// CHECK-DAG: %[[INIT1:.*]] = ortumcore.acc_init : !ortumcore.acc
// CHECK-DAG: %[[VALUE0:.*]] = vector.extract %[[VALUES]][0] : vector<2xi16>
// CHECK-DAG: %[[VALUE1:.*]] = vector.extract %[[VALUES]][1] : vector<2xi16>
// CHECK: %[[OUT0:.*]], %[[OUT1:.*]] = ortumcore.dmac %[[INIT0]], %[[INIT1]], %[[VALUE0]], %[[COEFFICIENT]], %[[VALUE1]], %[[COEFFICIENT]]
// CHECK: %[[LANE0:.*]] = ortumcore.acc_out %[[OUT0]] {shift = 15
// CHECK: vector.insert %[[LANE0]], {{.*}} [0]
// CHECK: %[[LANE1:.*]] = ortumcore.acc_out %[[OUT1]] {shift = 15
// CHECK: vector.insert %[[LANE1]], {{.*}} [1]
// CHECK-NOT: ondsp.

// -----

!pair = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 2>

// The pair threads scf.for iter_args as two target accumulators.
func.func @loop_carried(%values: memref<?xvector<2xi16>>, %coefficient: i16,
                        %count: index) -> vector<2xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = ondsp.acc_zero : !pair
  %acc = scf.for %i = %c0 to %count step %c1 iter_args(%it = %zero) -> (!pair) {
    %v = memref.load %values[%i] : memref<?xvector<2xi16>>
    %next = ondsp.mac %it, %v, %coefficient {
      numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
      product = #ondsp.product<full>
    } : (!pair, vector<2xi16>, i16) -> !pair
    scf.yield %next : !pair
  }
  %out = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i32, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<saturate>
  } : (!pair) -> vector<2xi32>
  return %out : vector<2xi32>
}

// CHECK-LABEL: func.func @loop_carried(
// CHECK: %[[INIT0:.*]] = ortumcore.acc_init
// CHECK: %[[INIT1:.*]] = ortumcore.acc_init
// CHECK: %[[LOOP:.*]]:2 = scf.for {{.*}} iter_args(%[[IT0:.*]] = %[[INIT0]], %[[IT1:.*]] = %[[INIT1]]) -> (!ortumcore.acc, !ortumcore.acc)
// CHECK: %[[STEP0:.*]], %[[STEP1:.*]] = ortumcore.dmac %[[IT0]], %[[IT1]],
// CHECK: scf.yield %[[STEP0]], %[[STEP1]]
// CHECK-DAG: ortumcore.acc_out %[[LOOP]]#0 {shift = 15
// CHECK-DAG: ortumcore.acc_out %[[LOOP]]#1 {shift = 15
// CHECK-NOT: ondsp.

// -----

!pair = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 2>

// A ties-positive pair export composes the proven increment-and-halve readout
// once per lane, exactly as the single-lane conversion does.
func.func @ntp_export(%values: vector<2xi16>, %coefficient: i16) -> vector<2xi16> {
  %zero = ondsp.acc_zero : !pair
  %acc = ondsp.mac %zero, %values, %coefficient {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!pair, vector<2xi16>, i16) -> !pair
  %out = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_ties_positive>,
    overflow = #ondsp.overflow<saturate>
  } : (!pair) -> vector<2xi16>
  return %out : vector<2xi16>
}

// CHECK-LABEL: func.func @ntp_export(
// CHECK: ortumcore.dmac
// CHECK-COUNT-2: ortumcore.acc_out {{.*}} {shift = 14
// CHECK: return {{.*}} : vector<2xi16>
// CHECK-NOT: ondsp.

// -----

!pair = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 2>

// nearest_even is outside the readout capability, so the eligibility scan
// leaves the whole function for the fail-closed main conversion.
func.func @keeps_unsupported_export(%values: vector<2xi16>, %coefficient: i16) -> vector<2xi32> {
  %zero = ondsp.acc_zero : !pair
  %acc = ondsp.mac %zero, %values, %coefficient {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!pair, vector<2xi16>, i16) -> !pair
  %out = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i32, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (!pair) -> vector<2xi32>
  return %out : vector<2xi32>
}

// CHECK-LABEL: func.func @keeps_unsupported_export(
// CHECK: ondsp.acc_zero
// CHECK: ondsp.mac
// CHECK: ondsp.acc_export
// CHECK-NOT: ortumcore.

// -----

!quad = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 4>

// Only pairs map onto the dual-lane capability; wider lane counts stay.
func.func @keeps_lanes_four(%values: vector<4xi16>, %coefficient: i16) -> vector<4xi32> {
  %zero = ondsp.acc_zero : !quad
  %acc = ondsp.mac %zero, %values, %coefficient {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!quad, vector<4xi16>, i16) -> !quad
  %out = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i32, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<saturate>
  } : (!quad) -> vector<4xi32>
  return %out : vector<4xi32>
}

// CHECK-LABEL: func.func @keeps_lanes_four(
// CHECK: ondsp.mac
// CHECK-NOT: ortumcore.

// -----

!pair = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 2>

// A pair crossing the function boundary is not function-local; the scan
// refuses and the main conversion reports it.
func.func @keeps_signature_pair(%acc: !pair) -> vector<2xi32> {
  %out = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i32, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<saturate>
  } : (!pair) -> vector<2xi32>
  return %out : vector<2xi32>
}

// CHECK-LABEL: func.func @keeps_signature_pair(
// CHECK: ondsp.acc_export
// CHECK-NOT: ortumcore.

// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --cse --canonicalize --vectorize-ondsp-fixed-decimate-outputs="vector-width=8" > %t.batched.mlir
// RUN: ondrix-opt %t.batched.mlir --vectorize-ondsp-fixed-memref-reduce="vector-width=4" --parallelize-ondsp-fixed-wrap-vector-reduce --normalize-ondsp-fixed-vector-reduce | FileCheck %s

// The two vectorization axes are orthogonal and must stay that way.
//
// Vertical batching spends vector lanes on independent OUTPUTS: no lane is
// combined with another, so it is order preserving and needs no proof. The
// horizontal reduction passes spend lanes on the REDUCTION axis: they change
// the fold order and therefore need a range or reassociation proof. If a single
// accumulator value carried both meanings the two readings would collide, so a
// multi-lane accumulator is refused by `reduce_mac` itself and by the shared
// fixed Vector MAC domain gate the horizontal passes consult.
//
// What that buys, pinned here: running the horizontal passes after batching
// leaves the batched body exactly as it was and chunks only the ordered
// remainder.

// CHECK-LABEL: func.func @batched_then_horizontal
// The batched block is untouched: still one span load and one even-lane shuffle
// per tap, still a multi-lane `mac` against a scalar coefficient, still a
// multi-lane export. No horizontal reduction appeared inside it.
// CHECK: scf.for
// CHECK-COUNT-8: vector.load {{.*}} : memref<44xi16>, vector<16xi16>
// CHECK-NOT: vector.reduction
// CHECK-NOT: ondsp.acc_add_term
// CHECK: ondsp.acc_export
// CHECK-SAME: update_overflow = saturate, lanes = 8>) -> vector<8xi16>
// CHECK: vector.store {{.*}} : memref<19xi16>, vector<8xi16>

// The ordered remainder is the one the horizontal passes act on: its reduction
// becomes four-lane product chunks folded in increasing lane order by explicit
// single-lane accumulator terms, plus a scalar tail.
// CHECK: scf.for
// CHECK: ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate>
// CHECK: arith.muli {{.*}} : vector<4xi32>
// CHECK-COUNT-4: ondsp.acc_add_term
// CHECK-SAME: (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i32)
// CHECK: ondsp.mac
// CHECK-SAME: (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16)
// CHECK-NOT: ondsp.reduce_mac

func.func @batched_then_horizontal(
    %input: tensor<44xi16>, %coeffs: tensor<8xi16>, %init: tensor<19xi16>) -> tensor<19xi16> {
  %result = ondrix.fir_decimate %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    factor = 2,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<44xi16>, tensor<8xi16>, tensor<19xi16>) -> tensor<19xi16>
  return %result : tensor<19xi16>
}

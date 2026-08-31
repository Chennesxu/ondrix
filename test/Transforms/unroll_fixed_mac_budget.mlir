// RUN: ondrix-opt %s --unroll-ondsp-fixed-mac-loops=max-unrolled-terms=16 | FileCheck %s --check-prefix=ADMIT
// RUN: ondrix-opt %s --unroll-ondsp-fixed-mac-loops=max-unrolled-terms=15 | FileCheck %s --check-prefix=REFUSE
// RUN: ondrix-opt %s --scalarize-ondsp-fixed-reduce-mac=max-unrolled-terms=8 --unroll-ondsp-fixed-mac-loops=max-unrolled-terms=8 | FileCheck %s --check-prefix=COMBINED
// RUN: ondrix-opt %s --scalarize-ondsp-fixed-reduce-mac=max-unrolled-terms=16 --unroll-ondsp-fixed-mac-loops=max-unrolled-terms=16 | FileCheck %s --check-prefix=PAIRED

// The budget prices what unrolling PRODUCES. A nested candidate multiplies
// rather than adds, so this pair costs 4 x 4, not the 4 + 4 a trip sum claims.
// ADMIT-LABEL: func.func @nested_pair
// ADMIT-NOT: scf.for
// ADMIT-COUNT-16: ondsp.mac
// REFUSE-LABEL: func.func @nested_pair
// REFUSE: scf.for
// REFUSE: scf.for
func.func @nested_pair(%lhs: memref<64xi16>, %rhs: memref<64xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %seed = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %outer = scf.for %i = %c0 to %c4 step %c1 iter_args(%oa = %seed) -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) {
    %inner = scf.for %j = %c0 to %c4 step %c1 iter_args(%ia = %oa) -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) {
      %a = memref.load %lhs[%j] : memref<64xi16>
      %b = memref.load %rhs[%j] : memref<64xi16>
      %m = ondsp.mac %ia, %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
      scf.yield %m : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
    }
    scf.yield %inner : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  }
  return %outer : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
}

// Neither pass may price this at four on its own: scalarize expands the
// reduction in place, then the trip multiplies what it left in the body.
// COMBINED-LABEL: func.func @scalarize_then_unroll
// COMBINED: scf.for
// COMBINED-COUNT-4: ondsp.mac
// COMBINED-NOT: ondsp.mac
func.func @scalarize_then_unroll(%lhs: memref<4xi16>, %rhs: memref<4xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %seed = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %outer = scf.for %i = %c0 to %c4 step %c1 iter_args(%oa = %seed) -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) {
    %r = ondsp.reduce_mac %oa, %lhs, %rhs {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, memref<4xi16>, memref<4xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
    scf.yield %r : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  }
  return %outer : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
}

// A candidate reached through a region op still multiplies: pricing the
// enclosing op by its raw mac count reads this 4 x 4 nest as 4.
// ADMIT-LABEL: func.func @nested_behind_region_op
// ADMIT-NOT: scf.for
// ADMIT-COUNT-16: ondsp.mac
// REFUSE-LABEL: func.func @nested_behind_region_op
// REFUSE: scf.for
// REFUSE: scf.for
// COMBINED-LABEL: func.func @nested_behind_region_op
// COMBINED: scf.for
func.func @nested_behind_region_op(%lhs: memref<64xi16>, %rhs: memref<64xi16>, %p: i1) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %seed = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %outer = scf.for %i = %c0 to %c4 step %c1 iter_args(%oa = %seed) -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) {
    %g = scf.if %p -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) {
      %inner = scf.for %j = %c0 to %c4 step %c1 iter_args(%ia = %oa) -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) {
        %a = memref.load %lhs[%j] : memref<64xi16>
        %b = memref.load %rhs[%j] : memref<64xi16>
        %m = ondsp.mac %ia, %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
        scf.yield %m : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
      }
      scf.yield %inner : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
    } else {
      scf.yield %oa : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
    }
    scf.yield %g : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  }
  return %outer : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
}

// The scalarizer recovers a static extent through the casts bufferization
// inserts, so this function costs sixteen and both reductions are declined.
// Unroll must recover the same extent from the memref.dim bound the refused
// path emits, or it re-expands the direct one and mixes the two shapes.
// COMBINED-LABEL: func.func @direct_and_cast_reductions
// COMBINED-COUNT-2: ondsp.mac
// COMBINED-NOT: ondsp.mac
// PAIRED-LABEL: func.func @direct_and_cast_reductions
// PAIRED-NOT: scf.for
// PAIRED-COUNT-16: ondsp.mac
func.func @direct_and_cast_reductions(%l: memref<8xi16>, %r: memref<8xi16>) -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) {
  %s = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %direct = ondsp.reduce_mac %s, %l, %r {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, memref<8xi16>, memref<8xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %lc = memref.cast %l : memref<8xi16> to memref<?xi16>
  %rc = memref.cast %r : memref<8xi16> to memref<?xi16>
  %cast = ondsp.reduce_mac %s, %lc, %rc {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, memref<?xi16>, memref<?xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  return %direct, %cast : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
}

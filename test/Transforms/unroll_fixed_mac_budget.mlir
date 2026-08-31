// RUN: ondrix-opt %s --unroll-ondsp-fixed-mac-loops=max-unrolled-terms=1024 | FileCheck %s --check-prefix=WIDE
// RUN: ondrix-opt %s --scalarize-ondsp-fixed-reduce-mac=max-unrolled-terms=8 --unroll-ondsp-fixed-mac-loops=max-unrolled-terms=8 | FileCheck %s --check-prefix=TIGHT
// RUN: ondrix-opt %s --scalarize-ondsp-fixed-reduce-mac=max-unrolled-terms=16 --unroll-ondsp-fixed-mac-loops=max-unrolled-terms=16 | FileCheck %s --check-prefix=ROOMY

// A loop carrying another loop is refused rather than priced, at any budget:
// pricing it means modelling what unrolling does inside the nest, and nothing
// lowers to that shape here. A guard region is priced and still unrolls.
// WIDE-LABEL: func.func @nested_pair
// WIDE: scf.for
// WIDE: scf.for
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

// The nest is still a nest behind a region op, so the refusal must see through
// one: pricing the enclosing op by its own macs reads this 4 x 4 as 4.
// WIDE-LABEL: func.func @nested_behind_region_op
// WIDE: scf.for
// WIDE: scf.for
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

// Neither pass may price this at four on its own: scalarize expands the
// reduction in place, then the trip multiplies what it left in the body.
// TIGHT-LABEL: func.func @scalarize_then_unroll
// TIGHT: scf.for
// TIGHT-COUNT-4: ondsp.mac
// TIGHT-NOT: ondsp.mac
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

// The scalarizer recovers a static extent through the casts bufferization
// inserts and prices this function at sixteen, declining both reductions. The
// loop it leaves for the cast one is priced here too, or the direct one is
// expanded alone and the function mixes both shapes.
// TIGHT-LABEL: func.func @direct_and_cast_reductions
// TIGHT-COUNT-2: ondsp.mac
// TIGHT-NOT: ondsp.mac
// ROOMY-LABEL: func.func @direct_and_cast_reductions
// ROOMY-NOT: scf.for
// ROOMY-COUNT-16: ondsp.mac
func.func @direct_and_cast_reductions(%l: memref<8xi16>, %r: memref<8xi16>) -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) {
  %s = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %direct = ondsp.reduce_mac %s, %l, %r {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, memref<8xi16>, memref<8xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %lc = memref.cast %l : memref<8xi16> to memref<?xi16>
  %rc = memref.cast %r : memref<8xi16> to memref<?xi16>
  %cast = ondsp.reduce_mac %s, %lc, %rc {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, memref<?xi16>, memref<?xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  return %direct, %cast : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
}

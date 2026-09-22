// RUN: ondrix-opt %s --ondrix-default-pipeline="vector-bits=0" --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=NOBLOCK
// RUN: ondrix-opt %s --ondrix-default-pipeline="vector-bits=0 hardware-repeat-block=true" --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=BLOCK
// RUN: ondrix-opt %s --ondrix-default-pipeline="vector-bits=0 hardware-repeat-block=true" | FileCheck %s --check-prefix=KEPT

// The straight-line reduction form pays for itself by deleting the index
// update and the branch. A declared repeat block has already deleted both, so
// the budgets drop to one term and the counted loop survives for it to claim.
// That is a blunt instrument -- a reduction carrying nothing across a loop
// still wins unrolled -- which is what `max-carried-window` exists to refine;
// it stays 0 here because no target has declared one.

// NOBLOCK: scalarize-ondsp-certified-constant-reduce{max-carried-window=0 max-elements=256 max-unrolled-terms=128 scalar-register-bits=32}
// NOBLOCK: scalarize-ondsp-fixed-reduce-mac{max-carried-window=0 max-unrolled-terms=128}
// NOBLOCK: unroll-ondsp-fixed-mac-loops{max-carried-window=0 max-unrolled-terms=128}
// NOBLOCK: unroll-ondsp-fp-ordered-reduce{{.*}}max-straight-line-terms=256 max-unrolled-terms=512
// BLOCK: scalarize-ondsp-certified-constant-reduce{max-carried-window=0 max-elements=256 max-unrolled-terms=1 scalar-register-bits=32}
// BLOCK: scalarize-ondsp-fixed-reduce-mac{max-carried-window=0 max-unrolled-terms=1}
// BLOCK: unroll-ondsp-fixed-mac-loops{max-carried-window=0 max-unrolled-terms=1}
// BLOCK: unroll-ondsp-fp-ordered-reduce{{.*}}max-straight-line-terms=1 max-unrolled-terms=1

// The budget is not bookkeeping: under it the 64 taps stay one counted loop
// rather than 64 straight-line multiply-adds.
// KEPT-LABEL: llvm.func @dot64_q15
// KEPT: llvm.cond_br
// KEPT-COUNT-1: llvm.mul
// KEPT-NOT: llvm.mul

func.func @dot64_q15(%lhs: memref<64xi16>, %rhs: memref<64xi16>) -> i16 {
  %init = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %r = ondsp.reduce_mac %init, %lhs, %rhs {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, memref<64xi16>, memref<64xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %out = ondsp.acc_export %r {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, rounding = #ondsp.rounding<nearest_even>, overflow = #ondsp.overflow<saturate>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) -> i16
  return %out : i16
}

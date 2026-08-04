// RUN: ondrix-compile %S/Inputs/q15_call_forward.ox | FileCheck %s --check-prefix=FORWARD
// RUN: ondrix-compile %S/Inputs/q15_call_spectrum.ox | FileCheck %s --check-prefix=CHAIN
// RUN: not ondrix-compile %S/Inputs/invalid_call_arity.ox 2>&1 | FileCheck %s --check-prefix=ARITY
// RUN: not ondrix-compile %S/Inputs/invalid_call_shape.ox 2>&1 | FileCheck %s --check-prefix=SHAPE
// RUN: not ondrix-compile %S/Inputs/invalid_call_recursive.ox 2>&1 | FileCheck %s --check-prefix=RECURSIVE
// RUN: not ondrix-compile %S/Inputs/invalid_call_duplicate.ox 2>&1 | FileCheck %s --check-prefix=DUPLICATE
// RUN: not ondrix-compile %S/Inputs/invalid_callee_body.ox 2>&1 | FileCheck %s --check-prefix=CALLEE

// The module exports the last function only; a callee is a named body, and
// the contracts written in it are the ones that reach the operation.
// FORWARD-NOT: @prefilter
// FORWARD-LABEL: func.func @q15_call_forward(
// FORWARD: ondrix.fir_filter
// FORWARD-SAME: storage = i40
// FORWARD-SAME: update_overflow = saturate>

// A callee composes wherever a nested expression already composes: here the
// filter stage of an FFT chain, with its inferred accumulator intact.
// CHAIN-LABEL: func.func @q15_call_spectrum(
// CHAIN: ondrix.fir_filter
// CHAIN-SAME: storage = i35
// CHAIN: ondrix.rfft
// CHAIN: ondrix.cx_magnitude

// ARITY: error: 'helper' takes 2 arguments, but 1 were given
// SHAPE: error: argument 1 of 'helper' does not match the declared parameter 'x'
// RECURSIVE: error: function 'invalid_call_recursive' cannot call itself

// DUPLICATE: error: function 'helper' is already declared

// A callee is checked against its own signature, so a body that cannot hold
// its declaration is reported at the callee, not at the call.
// CALLEE: invalid_callee_body.ox:2:10: error: dct currently requires a power-of-two input extent

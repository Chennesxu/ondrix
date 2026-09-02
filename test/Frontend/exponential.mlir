// RUN: ondrix-compile %S/Inputs/q15_log2.ox | FileCheck %s --check-prefix=LOG
// RUN: ondrix-compile %S/Inputs/q15_exp2.ox | FileCheck %s --check-prefix=EXP
// RUN: ondrix-compile %S/Inputs/q31_log2.ox | FileCheck %s --check-prefix=LOG31
// RUN: ondrix-compile %S/Inputs/q31_exp2.ox | FileCheck %s --check-prefix=EXP31

// The source type system names only the i16 storage, so the two readings the
// contract distinguishes are supplied by the binding rather than spelled at
// the call site — the same projection dct's derived output frac uses.
// LOG-LABEL: func.func @q15_log2(
// LOG: ondrix.log2
// LOG-SAME: numeric = #ondsp.fixed<unsigned, storage = i16, frac = 16>
// LOG-SAME: output_numeric = #ondsp.fixed<signed, storage = i16, frac = 11>

// EXP-LABEL: func.func @q15_exp2(
// EXP: ondrix.exp2
// EXP-SAME: numeric = #ondsp.fixed<signed, storage = i16, frac = 11>
// EXP-SAME: output_numeric = #ondsp.fixed<unsigned, storage = i16, frac = 16>

// At q31 the same projection supplies the Q0.32 magnitude and Q6.26 exponent.
// LOG31-LABEL: func.func @q31_log2(
// LOG31: ondrix.log2
// LOG31-SAME: numeric = #ondsp.fixed<unsigned, storage = i32, frac = 32>
// LOG31-SAME: output_numeric = #ondsp.fixed<signed, storage = i32, frac = 26>

// EXP31-LABEL: func.func @q31_exp2(
// EXP31: ondrix.exp2
// EXP31-SAME: numeric = #ondsp.fixed<signed, storage = i32, frac = 26>
// EXP31-SAME: output_numeric = #ondsp.fixed<unsigned, storage = i32, frac = 32>

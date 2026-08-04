// RUN: ondrix-compile %S/Inputs/q15_log2.ox | FileCheck %s --check-prefix=LOG
// RUN: ondrix-compile %S/Inputs/q15_exp2.ox | FileCheck %s --check-prefix=EXP

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

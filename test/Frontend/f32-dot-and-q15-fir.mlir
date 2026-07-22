// RUN: ondrix-compile %S/Inputs/f32_dot_fma.ox | FileCheck %s --check-prefix=F32
// RUN: ondrix-compile %S/Inputs/f32_dot_fast.ox | FileCheck %s --check-prefix=FAST
// RUN: ondrix-compile %S/Inputs/f32_dot_fma.ox | ondrix-opt --convert-ondrix-to-ondsp --lower-ondsp-f32-reduce-to-scalar | FileCheck %s --check-prefix=F32-SCALAR
// RUN: ondrix-compile %S/Inputs/q15_fir.ox | FileCheck %s --check-prefix=FIR
// RUN: ondrix-compile %S/Inputs/q15_fir.ox | ondrix-opt --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar | FileCheck %s --check-prefix=FIR-SCALAR

// F32-LABEL: func.func @f32_dot_fma
// F32-SAME: memref<?xf32>
// F32-SAME: memref<?xf32>
// F32-SAME: -> f32
// F32: ondrix.dot
// F32-SAME: numeric = #ondsp.fp<format = f32, contract = fma>
// F32-NOT: ondsp.acc_export

// FAST-LABEL: func.func @f32_dot_fast
// FAST: numeric = #ondsp.fp<format = f32, contract = fast>

// F32-SCALAR-LABEL: func.func @f32_dot_fma
// F32-SCALAR: cf.assert
// F32-SCALAR: scf.for
// F32-SCALAR: math.fma
// F32-SCALAR-NOT: ondrix.
// F32-SCALAR-NOT: ondsp.reduce_mac

// FIR-LABEL: func.func @q15_fir
// FIR: %[[ACC:.*]] = ondrix.fir
// FIR-SAME: numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
// FIR: ondsp.acc_export %[[ACC]]

// FIR-SCALAR-LABEL: func.func @q15_fir
// FIR-SCALAR: cf.assert
// FIR-SCALAR: scf.for
// FIR-SCALAR: arith.muli
// FIR-SCALAR-NOT: ondrix.
// FIR-SCALAR-NOT: ondsp.

// RUN: ondrix-compile %S/Inputs/f32_dot_fma.ox | FileCheck %s --check-prefix=F32
// RUN: ondrix-compile %S/Inputs/f32_dot_fast.ox | FileCheck %s --check-prefix=FAST
// RUN: ondrix-compile %S/Inputs/f32_dot_fma.ox | ondrix-opt --convert-ondrix-to-ondsp --lower-ondsp-f32-reduce-to-scalar | FileCheck %s --check-prefix=F32-SCALAR
// RUN: ondrix-compile %S/Inputs/q15_fir.ox | FileCheck %s --check-prefix=FIR
// RUN: ondrix-compile %S/Inputs/q15_fir.ox | ondrix-opt --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar | FileCheck %s --check-prefix=FIR-SCALAR
// RUN: ondrix-compile %S/Inputs/q15_fir_constexpr.ox | FileCheck %s --check-prefix=CONSTEXPR
// RUN: ondrix-compile %S/Inputs/q15_fir_constexpr.ox | ondrix-opt --specialize-ondrix-constant-fir | FileCheck %s --check-prefix=SPECIALIZED
// RUN: ondrix-compile %S/Inputs/f32_fir_fma.ox | FileCheck %s --check-prefix=F32-FIR
// RUN: ondrix-compile %S/Inputs/f32_fir_fma.ox | ondrix-opt --convert-ondrix-to-ondsp --lower-ondsp-f32-reduce-to-scalar | FileCheck %s --check-prefix=F32-FIR-SCALAR
// RUN: ondrix-compile %S/Inputs/q15_dot_constexpr.ox | FileCheck %s --check-prefix=CONST-DOT
// RUN: ondrix-compile %S/Inputs/q15_dot_constexpr.ox | ondrix-opt --convert-ondrix-to-ondsp --vectorize-ondsp-constant-saturating-memref-reduce="vector-width=4 max-elements=64" | FileCheck %s --check-prefix=PROVEN-DOT

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

// CONSTEXPR: memref.global "private" constant @__ox_q15_fir_constexpr_coefficients
// CONSTEXPR-SAME: : memref<5xi16> = dense<[16384, -8192, 4096, -8192, 16384]>
// CONSTEXPR-LABEL: func.func @q15_fir_constexpr(
// CONSTEXPR-SAME: %[[WINDOW:.*]]: memref<5xi16>) -> i16
// CONSTEXPR: %[[COEFFS:.*]] = memref.get_global @__ox_q15_fir_constexpr_coefficients
// CONSTEXPR: ondrix.fir %[[WINDOW]], %[[COEFFS]]

// SPECIALIZED-LABEL: func.func @q15_fir_constexpr(
// SPECIALIZED-NOT: ondrix.fir
// SPECIALIZED: ondsp.acc_add_term

// F32-FIR-LABEL: func.func @f32_fir_fma
// F32-FIR: ondrix.fir
// F32-FIR-SAME: numeric = #ondsp.fp<format = f32, contract = fma>

// F32-FIR-SCALAR-LABEL: func.func @f32_fir_fma
// F32-FIR-SCALAR: cf.assert
// F32-FIR-SCALAR: scf.for
// F32-FIR-SCALAR: math.fma
// F32-FIR-SCALAR-NOT: ondrix.
// F32-FIR-SCALAR-NOT: ondsp.

// CONST-DOT: memref.global "private" constant @__ox_q15_dot_constexpr_rhs
// CONST-DOT-LABEL: func.func @q15_dot_constexpr(%{{.*}}: memref<8xi16>) -> i16
// CONST-DOT: ondrix.dot

// PROVEN-DOT-LABEL: func.func @q15_dot_constexpr
// PROVEN-DOT: vector.reduction <add>, {{.*}} : vector<4xi64> into i64
// PROVEN-DOT: ondsp.acc_add_term
// PROVEN-DOT-NOT: ondsp.reduce_mac

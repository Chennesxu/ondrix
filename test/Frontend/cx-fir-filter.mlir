// RUN: ondrix-compile %S/Inputs/q15_cx_fir.ox | FileCheck %s --check-prefix=FIR
// RUN: ondrix-compile %S/Inputs/q15_cx_fir.ox | ondrix-opt --convert-ondrix-to-ondsp --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --convert-ondsp-to-ortumcore | FileCheck %s --check-prefix=TARGET
// RUN: ondrix-compile %S/Inputs/q15_cx_matched_filter.ox | FileCheck %s --check-prefix=MATCHED
// RUN: not ondrix-compile %S/Inputs/invalid_cx_fir_boundary.ox 2>&1 | FileCheck %s --check-prefix=BOUNDARY
// RUN: not ondrix-compile %S/Inputs/invalid_cx_fir_extent.ox 2>&1 | FileCheck %s --check-prefix=EXTENT

// FIR-LABEL: func.func @q15_cx_fir
// FIR-SAME: tensor<10xi32>
// FIR-SAME: tensor<4xi32>
// FIR-SAME: -> tensor<7xi32>
// FIR: ondrix.cx_fir_filter
// FIR-SAME: !ondsp.acc<storage = i32, frac = 30, signed, update_overflow = saturate>
// FIR-NOT: conjugate

// The operation survives the ondsp conversion untouched: bufferization is its
// only lowering, and it reaches the target reduction through that.
// TARGET-LABEL: func.func @q15_cx_fir
// TARGET: scf.for
// TARGET: ortumcore.cx_reduce_mac
// TARGET-NOT: ondsp.
// TARGET-NOT: ondrix.

// MATCHED-LABEL: func.func @q15_cx_matched_filter
// MATCHED: ondrix.cx_fir_filter
// MATCHED-SAME: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
// MATCHED-SAME: conjugate

// BOUNDARY: error: cx_fir_filter currently supports boundary=valid
// EXTENT: error: valid-boundary cx_fir_filter requires input extent >= tap count

// RUN: ondrix-compile %S/Inputs/q15_cx_fir.ox | FileCheck %s --check-prefix=FIR
// RUN: ondrix-compile %S/Inputs/q15_cx_fir.ox | ondrix-opt --convert-ondrix-to-ondsp --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --convert-ondsp-to-ortumcore | FileCheck %s --check-prefix=TARGET
// RUN: ondrix-compile %S/Inputs/q15_cx_matched_filter.ox | FileCheck %s --check-prefix=MATCHED
// RUN: ondrix-compile %S/Inputs/q31_cx_fir.ox | FileCheck %s --check-prefix=Q31
// RUN: not ondrix-compile %S/Inputs/invalid_cx_fir_boundary.ox 2>&1 | FileCheck %s --check-prefix=BOUNDARY
// RUN: not ondrix-compile %S/Inputs/invalid_cx_fir_extent.ox 2>&1 | FileCheck %s --check-prefix=EXTENT
// RUN: ondrix-compile %S/Inputs/f32_cx_fir.ox | FileCheck %s --check-prefix=F32
// RUN: not ondrix-compile %S/Inputs/invalid_f32_cx_fir_accumulator.ox 2>&1 | FileCheck %s --check-prefix=F32ACC
// RUN: not ondrix-compile %S/Inputs/invalid_f32_cx_dot.ox 2>&1 | FileCheck %s --check-prefix=F32DOT

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

// Q31-LABEL: func.func @q31_cx_fir
// Q31-SAME: tensor<10xi64>
// Q31-SAME: -> tensor<7xi64>
// Q31: ondrix.cx_fir_filter
// Q31-SAME: !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>
// Q31-SAME: layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>

// BOUNDARY: error: cx_fir_filter currently supports boundary=valid
// EXTENT: error: valid-boundary cx_fir_filter requires input extent >= tap count

// The interleaved f32 sliding form: extents count complex values and the
// emitted tensors count elements, so every length doubles, and the window
// declares a contract mode where the packed widths declare an accumulator.
// F32-LABEL: func.func @f32_cx_fir
// F32-SAME: tensor<24xf32>
// F32-SAME: tensor<8xf32>
// F32-SAME: -> tensor<18xf32>
// F32: ondrix.cx_fir_filter
// F32-SAME: conjugate
// F32-SAME: layout = #ondsp.cx_layout<interleaved>
// F32-SAME: numeric = #ondsp.fp<format = f32, contract = fma>
// F32-NOT: accumulator

// F32ACC: error: expected floating-point contract policy

// The operation carries the profile; the SOURCE has no complex f32 scalar,
// and the diagnostic says which spelling does have one.
// F32DOT: error: cx_dot has no complex_f32 spelling

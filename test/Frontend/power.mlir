// RUN: ondrix-compile %S/Inputs/q15_power.ox | FileCheck %s --check-prefix=POWER
// RUN: ondrix-compile %S/Inputs/q15_power.ox | ondrix-opt --convert-ondrix-to-ondsp --convert-ondsp-to-ortumcore | FileCheck %s --check-prefix=TARGET
// RUN: ondrix-compile %S/Inputs/q15_power_spectrum.ox | FileCheck %s --check-prefix=SPECTRUM
// RUN: not ondrix-compile %S/Inputs/invalid_power_element.ox 2>&1 | FileCheck %s --check-prefix=ELEMENT
// RUN: ondrix-compile %S/Inputs/f32_power_spectrum.ox | FileCheck %s --check-prefix=F32
// RUN: not ondrix-compile %S/Inputs/invalid_f32_power_rounding.ox 2>&1 | FileCheck %s --check-prefix=F32ROUND

// The declared result width names the reading; the source spells only the
// tie rule, and the shift follows from the two fractions.
// POWER-LABEL: func.func @q15_power
// POWER-SAME: tensor<8xi32>
// POWER-SAME: -> tensor<8xi16>
// POWER: ondrix.cx_power
// POWER-SAME: numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
// POWER-SAME: output_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
// POWER-SAME: rounding = #ondsp.rounding<toward_negative>

// TARGET-LABEL: func.func @q15_power
// TARGET: ortumcore.cx_power
// TARGET-SAME: shift = 15
// TARGET-NOT: ondsp.

// The power spectrum is the composition, with the transform's own boundary
// left intact between them.
// SPECTRUM-LABEL: func.func @q15_power_spectrum
// SPECTRUM: ondrix.rfft
// SPECTRUM: ondrix.cx_power
// SPECTRUM-SAME: rounding = #ondsp.rounding<nearest_ties_positive>

// ELEMENT: error: power requires complex_q15 or complex_f32 operand elements

// The f32 spectrum has a consumer: one bin is two elements, so the readout is
// half the transform's length and declares no reading at all.
// F32-LABEL: func.func @f32_power_spectrum
// F32: ondrix.rfft
// F32-SAME: (tensor<64xf32>) -> tensor<66xf32>
// F32: ondrix.cx_power
// F32-SAME: layout = #ondsp.cx_layout<interleaved>
// F32-SAME: numeric = #ondsp.fp<format = f32, contract = off>
// F32-SAME: (tensor<66xf32>) -> tensor<33xf32>
// F32-NOT: output_numeric

// F32ROUND: error: an f32 squared magnitude has no boundary to round

// RUN: ondrix-compile %S/Inputs/q15_power.ox | FileCheck %s --check-prefix=POWER
// RUN: ondrix-compile %S/Inputs/q15_power.ox | ondrix-opt --convert-ondrix-to-ondsp --convert-ondsp-to-ortumcore | FileCheck %s --check-prefix=TARGET
// RUN: ondrix-compile %S/Inputs/q15_power_spectrum.ox | FileCheck %s --check-prefix=SPECTRUM
// RUN: not ondrix-compile %S/Inputs/invalid_power_element.ox 2>&1 | FileCheck %s --check-prefix=ELEMENT

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

// ELEMENT: error: power requires complex_q15 operand elements

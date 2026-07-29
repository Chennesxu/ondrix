// RUN: ondrix-compile %S/Inputs/q15_dot.ox --print-source-locations | FileCheck %s --check-prefix=MLIR
// RUN: ondrix-compile %S/Inputs/q15_dot.ox | ondrix-opt --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar | FileCheck %s --check-prefix=SCALAR
// RUN: not ondrix-compile %S/Inputs/invalid_dot_ties_positive.ox 2>&1 | FileCheck %s --check-prefix=TIES

// The generic export policy exposes the three established tie rules. A
// newly declared dialect mode is opted into per builtin together with its
// contract and evidence, so it is refused here.
// TIES: invalid_dot_ties_positive.ox:2:10: error: export rounding must be nearest_even, toward_negative, or toward_zero

// MLIR: #loc = loc({{.*}}q15_dot.ox":1:1)
// MLIR-LABEL: func.func @q15_dot
// MLIR-SAME: %[[LHS:[^:]+]]: memref<?xi16>
// MLIR-SAME: %[[RHS:[^:]+]]: memref<?xi16>
// MLIR-SAME: -> i16
// MLIR-SAME: attributes {llvm.emit_c_interface}
// MLIR: %[[ACC:.*]] = ondrix.dot %[[LHS]], %[[RHS]]
// MLIR-SAME: numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
// MLIR-SAME: product = #ondsp.product<full>
// MLIR-SAME: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
// MLIR: %[[RESULT:.*]] = ondsp.acc_export %[[ACC]]
// MLIR-SAME: overflow = #ondsp.overflow<saturate>
// MLIR-SAME: rounding = #ondsp.rounding<nearest_even>
// MLIR: return %[[RESULT]] : i16
// MLIR: loc({{.*}}q15_dot.ox":2:10)

// SCALAR-LABEL: func.func @q15_dot
// SCALAR: cf.assert
// SCALAR: scf.for
// SCALAR: arith.muli
// SCALAR-NOT: ondrix.
// SCALAR-NOT: ondsp.

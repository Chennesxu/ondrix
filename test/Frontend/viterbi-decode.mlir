// RUN: ondrix-compile %S/Inputs/q15_viterbi_decode.ox | FileCheck %s --check-prefix=SOURCE
// RUN: ondrix-compile %S/Inputs/q15_viterbi_decode.ox --emit=c-header | FileCheck %s --check-prefix=HEADER
// RUN: ondrix-compile %S/Inputs/q15_viterbi_decode_r3.ox | FileCheck %s --check-prefix=RATE3
// RUN: not ondrix-compile %S/Inputs/invalid_viterbi_constraint.ox 2>&1 | FileCheck %s --check-prefix=CONSTRAINT
// RUN: not ondrix-compile %S/Inputs/invalid_viterbi_polynomial.ox 2>&1 | FileCheck %s --check-prefix=POLYNOMIAL
// RUN: not ondrix-compile %S/Inputs/invalid_viterbi_frame.ox 2>&1 | FileCheck %s --check-prefix=FRAME
// RUN: not ondrix-compile %S/Inputs/invalid_viterbi_q31.ox 2>&1 | FileCheck %s --check-prefix=WIDTH
// RUN: not ondrix-compile %S/Inputs/invalid_viterbi_result.ox 2>&1 | FileCheck %s --check-prefix=RESULT
// RUN: not ondrix-compile %S/Inputs/invalid_viterbi_nested.ox 2>&1 | FileCheck %s --check-prefix=NESTED
// RUN: not ondrix-compile %S/Inputs/invalid_u8_parameter.ox 2>&1 | FileCheck %s --check-prefix=U8PARAM
// RUN: not ondrix-compile %S/Inputs/invalid_u8_result.ox 2>&1 | FileCheck %s --check-prefix=U8RESULT

// Octal generators read as the integers they name; the packed bytes are u8.
// SOURCE: ondrix.viterbi_decode %arg0 {constraint_length = 7 : i64, polynomials = array<i64: 121, 91>} : (tensor<512xi16>) -> tensor<32xi8>
// HEADER: void ondrix_viterbi_k7(const int16_t *symbols, int8_t *output);
// RATE3: polynomials = array<i64: 21, 27, 31>} : (tensor<192xi16>) -> tensor<8xi8>

// CONSTRAINT: error: viterbi_decode constraint_length must be in [3, 7]
// POLYNOMIAL: error: every generator polynomial must lie in [1, 2^constraint_length) = [1, 128)
// FRAME: error: viterbi_decode requires N * R symbols with N a positive multiple of 8 and N * R <= 16384
// WIDTH: error: viterbi_decode requires q15 soft symbols
// RESULT: error: declared result type does not match the builtin expression
// NESTED: error: u8 decoded bits can only be returned
// U8PARAM: error: u8 is currently only the result of viterbi_decode
// U8RESULT: error: u8 is currently only the result of viterbi_decode

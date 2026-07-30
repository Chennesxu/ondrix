// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" | FileCheck %s

// The one admissible collision: a pre-declared private constant global whose
// initializer equals the required row bit for bit is REUSED rather than
// refused. Successful bufferization is itself part of the check — emitting a
// second definition would be a duplicate-symbol verifier failure — and the
// row-0 reduction must read the pre-declared table through get_global.

memref.global "private" constant @__ondrix_dct8_row0 : memref<8xi16> = dense<32767>

// The freshly created rows 1..7 are prepended at the module start, so they
// print BEFORE the pre-declared row 0; order is not part of the contract.
// CHECK-DAG: memref.global "private" constant @__ondrix_dct8_row0 : memref<8xi16> = dense<32767>
// CHECK-DAG: memref.global "private" constant @__ondrix_dct8_row1
// CHECK-LABEL: func.func @dct8_predeclared_q15(
// CHECK: memref.get_global @__ondrix_dct8_row0 : memref<8xi16>
// CHECK: ondsp.reduce_mac
func.func @dct8_predeclared_q15(%input: tensor<8xi16>) -> tensor<8xi16> {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 11>
  } : (tensor<8xi16>) -> tensor<8xi16>
  return %result : tensor<8xi16>
}

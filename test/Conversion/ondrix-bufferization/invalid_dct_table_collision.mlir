// RUN: not ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --split-input-file 2>&1 | FileCheck %s

// The reserved __ondrix_dct{N}_row{k} names are proof-carrying: the
// bufferization consumer feeds whatever resolves under them into a
// reduce_mac whose vector legality is then CORRECTLY proven for whatever
// contents are found. Reuse is therefore admitted only for a private
// constant memref.global of the exact type whose initializer equals the
// required coefficients bit for bit; every other occupant of the name must
// fail closed, because a content mismatch here compiles to a wrong-answer
// object, not to a diagnostic anywhere downstream.

// A same-named, same-type, private constant global with WRONG contents: the
// all-zero table. Without the initializer comparison this would silently
// pin the DC output to zero and the prefix proof would still succeed.
memref.global "private" constant @__ondrix_dct8_row0 : memref<8xi16> = dense<0>

// CHECK: error: 'ondrix.dct' op a foreign symbol occupies the reserved DCT coefficient table name or carries contents that differ from the required coefficients
func.func @dct8_zero_table(%input: tensor<8xi16>) -> tensor<8xi16> {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 11>
  } : (tensor<8xi16>) -> tensor<8xi16>
  return %result : tensor<8xi16>
}

// -----

// A single coefficient off by one LSB: the strongest content test, since
// every coarser structural check (kind, constness, type, initializer
// presence) passes.
memref.global "private" constant @__ondrix_dct8_row0 : memref<8xi16> = dense<[32767, 32767, 32767, 32767, 32767, 32767, 32766, 32767]>

// CHECK: error: 'ondrix.dct' op a foreign symbol occupies the reserved DCT coefficient table name or carries contents that differ from the required coefficients
func.func @dct8_one_coefficient_off(%input: tensor<8xi16>) -> tensor<8xi16> {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 11>
  } : (tensor<8xi16>) -> tensor<8xi16>
  return %result : tensor<8xi16>
}

// -----

// Correct contents but MUTABLE: a later store could rewrite the table after
// the proof ran, so constness is required even when the bits match today.
memref.global "private" @__ondrix_dct8_row0 : memref<8xi16> = dense<32767>

// CHECK: error: 'ondrix.dct' op a foreign symbol occupies the reserved DCT coefficient table name or carries contents that differ from the required coefficients
func.func @dct8_mutable_table(%input: tensor<8xi16>) -> tensor<8xi16> {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 11>
  } : (tensor<8xi16>) -> tensor<8xi16>
  return %result : tensor<8xi16>
}

// -----

// An external declaration: public visibility and no initializer, so the
// actual contents are a link-time unknown.
memref.global @__ondrix_dct8_row0 : memref<8xi16>

// CHECK: error: 'ondrix.dct' op a foreign symbol occupies the reserved DCT coefficient table name or carries contents that differ from the required coefficients
func.func @dct8_external_table(%input: tensor<8xi16>) -> tensor<8xi16> {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 11>
  } : (tensor<8xi16>) -> tensor<8xi16>
  return %result : tensor<8xi16>
}

// -----

// A symbol of a different kind entirely.
func.func private @__ondrix_dct8_row0() -> ()

// CHECK: error: 'ondrix.dct' op a foreign symbol occupies the reserved DCT coefficient table name or carries contents that differ from the required coefficients
func.func @dct8_function_symbol(%input: tensor<8xi16>) -> tensor<8xi16> {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 11>
  } : (tensor<8xi16>) -> tensor<8xi16>
  return %result : tensor<8xi16>
}

// -----

// The wrong memref type: right kind, right name, wrong shape.
memref.global "private" constant @__ondrix_dct8_row0 : memref<16xi16> = dense<32767>

// CHECK: error: 'ondrix.dct' op a foreign symbol occupies the reserved DCT coefficient table name or carries contents that differ from the required coefficients
func.func @dct8_wrong_type_table(%input: tensor<8xi16>) -> tensor<8xi16> {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 11>
  } : (tensor<8xi16>) -> tensor<8xi16>
  return %result : tensor<8xi16>
}

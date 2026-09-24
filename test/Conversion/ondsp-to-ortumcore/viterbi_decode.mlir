// RUN: ondrix-opt %s --split-input-file --convert-ondsp-to-ortumcore --verify-diagnostics | FileCheck %s

// A certified 16-bit K = 7 rate-1/2 decode whose generators tap both ends.
// CHECK-LABEL: func.func @k7_narrow
// CHECK: ortumcore.viterbi_decode %arg0, %arg1 {polynomials = array<i64: 121, 91>, renormalization_period = 117 : i64, symbol_bound = 127 : i64, unreachable_metric = -31244 : i64}
func.func @k7_narrow(%symbols: memref<512xi16>, %bits: memref<32xi8>) {
  ondsp.viterbi_decode %symbols, %bits {constraint_length = 7 : i64,
      polynomials = array<i64: 121, 91>, metric_bits = 16 : i64, symbol_bound = 127 : i64,
      unreachable_metric = -31244 : i64, renormalization_period = 117 : i64}
      : memref<512xi16>, memref<32xi8>
  return
}

// -----

func.func @k7_wide(%symbols: memref<128xi16>, %bits: memref<8xi8>) {
  // expected-error @+2 {{the target trellis carries 16-bit metrics, which a declared symbol_bound must certify first}}
  // expected-error @+1 {{failed to legalize operation 'ondsp.viterbi_decode'}}
  ondsp.viterbi_decode %symbols, %bits {constraint_length = 7 : i64,
      polynomials = array<i64: 121, 91>, metric_bits = 32 : i64, symbol_bound = 32768 : i64,
      unreachable_metric = -1073741825 : i64, renormalization_period = 0 : i64}
      : memref<128xi16>, memref<8xi8>
  return
}

// -----

func.func @k5_narrow(%symbols: memref<128xi16>, %bits: memref<8xi8>) {
  // expected-error @+2 {{the target trellis decodes constraint length 7 at rate 1/2}}
  // expected-error @+1 {{failed to legalize operation 'ondsp.viterbi_decode'}}
  ondsp.viterbi_decode %symbols, %bits {constraint_length = 5 : i64,
      polynomials = array<i64: 25, 27>, metric_bits = 16 : i64, symbol_bound = 127 : i64,
      unreachable_metric = -31752 : i64, renormalization_period = 0 : i64}
      : memref<128xi16>, memref<8xi8>
  return
}

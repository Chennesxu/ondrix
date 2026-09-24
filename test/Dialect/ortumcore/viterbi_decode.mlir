// RUN: ondrix-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @trellis_k7
// CHECK: ortumcore.viterbi_decode %arg0, %arg1
// CHECK-SAME: renormalization_period = 117
func.func @trellis_k7(%symbols: memref<512xi16>, %bits: memref<32xi8>) {
  ortumcore.viterbi_decode %symbols, %bits {polynomials = array<i64: 121, 91>,
      symbol_bound = 127 : i64, unreachable_metric = -31244 : i64,
      renormalization_period = 117 : i64} : memref<512xi16>, memref<32xi8>
  return
}

// -----

// 0o132 leaves bit 0 clear, so its two predecessors' branches are not one
// metric and its negation.
func.func @generator_misses_an_end(%symbols: memref<128xi16>, %bits: memref<8xi8>) {
  // expected-error @+1 {{the trellis unit decodes rate-1/2 codes whose two generators tap both ends}}
  ortumcore.viterbi_decode %symbols, %bits {polynomials = array<i64: 121, 90>,
      symbol_bound = 127 : i64, unreachable_metric = -31244 : i64,
      renormalization_period = 0 : i64} : memref<128xi16>, memref<8xi8>
  return
}

// -----

func.func @not_exact(%symbols: memref<512xi16>, %bits: memref<32xi8>) {
  // expected-error @+1 {{is not an exact 16-bit realization of the frame}}
  ortumcore.viterbi_decode %symbols, %bits {polynomials = array<i64: 121, 91>,
      symbol_bound = 127 : i64, unreachable_metric = -31244 : i64,
      renormalization_period = 0 : i64} : memref<512xi16>, memref<32xi8>
  return
}

// RUN: ondrix-opt %s | ondrix-opt | FileCheck %s

// CHECK-LABEL: func.func @viterbi_k7_r2
// CHECK: ondrix.viterbi_decode
// CHECK-SAME: constraint_length = 7
// CHECK-SAME: polynomials = array<i64: 121, 91>
func.func @viterbi_k7_r2(%symbols: tensor<512xi16>) -> tensor<32xi8> {
  %bits = ondrix.viterbi_decode %symbols {constraint_length = 7 : i64,
      polynomials = array<i64: 121, 91>} : (tensor<512xi16>) -> tensor<32xi8>
  return %bits : tensor<32xi8>
}

// The frame bound is inclusive: N * R = 16384.
// CHECK-LABEL: func.func @viterbi_largest_frame
func.func @viterbi_largest_frame(%symbols: tensor<16384xi16>) -> tensor<1024xi8> {
  %bits = ondrix.viterbi_decode %symbols {constraint_length = 3 : i64,
      polynomials = array<i64: 7, 5>} : (tensor<16384xi16>) -> tensor<1024xi8>
  return %bits : tensor<1024xi8>
}

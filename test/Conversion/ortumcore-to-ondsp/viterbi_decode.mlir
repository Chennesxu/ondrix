// RUN: ondrix-opt %s --convert-ortumcore-to-ondsp-emulation | FileCheck %s

// The unit's decode is the 16-bit ondsp realization at constraint length 7.
// CHECK-LABEL: func.func @trellis
// CHECK: ondsp.viterbi_decode %arg0, %arg1 {constraint_length = 7 : i64, metric_bits = 16 : i64, polynomials = array<i64: 121, 91>, renormalization_period = 117 : i64, symbol_bound = 127 : i64, unreachable_metric = -31244 : i64}
func.func @trellis(%symbols: memref<512xi16>, %bits: memref<32xi8>) {
  ortumcore.viterbi_decode %symbols, %bits {polynomials = array<i64: 121, 91>,
      symbol_bound = 127 : i64, unreachable_metric = -31244 : i64,
      renormalization_period = 117 : i64} : memref<512xi16>, memref<32xi8>
  return
}

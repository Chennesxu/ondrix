// RUN: ondrix-opt %s | ondrix-opt | FileCheck %s

// The wide realization every admitted frame decodes exactly with.
// CHECK-LABEL: func.func @viterbi_wide
// CHECK: ondsp.viterbi_decode %arg0, %arg1
// CHECK-SAME: metric_bits = 32
func.func @viterbi_wide(%symbols: memref<16384xi16>, %bits: memref<1024xi8>) {
  ondsp.viterbi_decode %symbols, %bits {constraint_length = 3 : i64,
      polynomials = array<i64: 7, 5>, metric_bits = 32 : i64, symbol_bound = 32768 : i64,
      unreachable_metric = -1073741825 : i64, renormalization_period = 0 : i64}
      : memref<16384xi16>, memref<1024xi8>
  return
}

// Symbols within 127 at K = 7: 16-bit metrics renormalized every 117 stages,
// the longest period that stays exact, with the lowest seed that cannot wrap.
// CHECK-LABEL: func.func @viterbi_narrow
// CHECK: ondsp.assume_magnitude_bound %arg0 {bound = 127 : i64}
// CHECK: renormalization_period = 117
func.func @viterbi_narrow(%symbols: memref<512xi16>, %bits: memref<32xi8>) {
  %bounded = ondsp.assume_magnitude_bound %symbols {bound = 127 : i64} : memref<512xi16>
  ondsp.viterbi_decode %bounded, %bits {constraint_length = 7 : i64,
      polynomials = array<i64: 121, 91>, metric_bits = 16 : i64, symbol_bound = 127 : i64,
      unreachable_metric = -31244 : i64, renormalization_period = 117 : i64}
      : memref<512xi16>, memref<32xi8>
  return
}

// RUN: ondrix-opt %s --narrow-ondsp-viterbi-metrics | FileCheck %s

// 64 stages of symbols within 127 grow a metric by at most 64 * 254 < 2^15:
// 16 bits with no renormalization, seeded at -2^15 + 6 * 254.
// CHECK-LABEL: func.func @k7_short
// CHECK: metric_bits = 16 : i64, polynomials = array<i64: 121, 91>, renormalization_period = 0 : i64, symbol_bound = 127 : i64, unreachable_metric = -31244 : i64
func.func @k7_short(%symbols: memref<128xi16>, %bits: memref<8xi8>) {
  %bounded = ondsp.assume_magnitude_bound %symbols {bound = 127 : i64} : memref<128xi16>
  ondsp.viterbi_decode %bounded, %bits {constraint_length = 7 : i64,
      polynomials = array<i64: 121, 91>, metric_bits = 32 : i64, symbol_bound = 32768 : i64,
      unreachable_metric = -1073741825 : i64, renormalization_period = 0 : i64}
      : memref<128xi16>, memref<8xi8>
  return
}

// 256 stages outgrow the width: renormalize every 32767 / 254 - 12 = 117.
// CHECK-LABEL: func.func @k7_long
// CHECK: metric_bits = 16 : i64, polynomials = array<i64: 121, 91>, renormalization_period = 117 : i64
func.func @k7_long(%symbols: memref<512xi16>, %bits: memref<32xi8>) {
  %bounded = ondsp.assume_magnitude_bound %symbols {bound = 127 : i64} : memref<512xi16>
  ondsp.viterbi_decode %bounded, %bits {constraint_length = 7 : i64,
      polynomials = array<i64: 121, 91>, metric_bits = 32 : i64, symbol_bound = 32768 : i64,
      unreachable_metric = -1073741825 : i64, renormalization_period = 0 : i64}
      : memref<512xi16>, memref<32xi8>
  return
}

// Witness: 1000 needs 3 (K - 1) D = 36000 of a 32768 seed range, so the
// decoder keeps its wide realization; and so does one with no declared bound.
// CHECK-LABEL: func.func @too_loose
// CHECK: metric_bits = 32
// CHECK-LABEL: func.func @undeclared
// CHECK: metric_bits = 32
func.func @too_loose(%symbols: memref<128xi16>, %bits: memref<8xi8>) {
  %bounded = ondsp.assume_magnitude_bound %symbols {bound = 1000 : i64} : memref<128xi16>
  ondsp.viterbi_decode %bounded, %bits {constraint_length = 7 : i64,
      polynomials = array<i64: 121, 91>, metric_bits = 32 : i64, symbol_bound = 32768 : i64,
      unreachable_metric = -1073741825 : i64, renormalization_period = 0 : i64}
      : memref<128xi16>, memref<8xi8>
  return
}
func.func @undeclared(%symbols: memref<128xi16>, %bits: memref<8xi8>) {
  ondsp.viterbi_decode %symbols, %bits {constraint_length = 7 : i64,
      polynomials = array<i64: 121, 91>, metric_bits = 32 : i64, symbol_bound = 32768 : i64,
      unreachable_metric = -1073741825 : i64, renormalization_period = 0 : i64}
      : memref<128xi16>, memref<8xi8>
  return
}

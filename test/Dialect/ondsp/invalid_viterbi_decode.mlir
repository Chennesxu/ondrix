// RUN: ondrix-opt %s --split-input-file --verify-diagnostics

// One stage past the longest exact period: 2 * 6 * 254 + 118 * 254 > 32767.
func.func @period_too_long(%symbols: memref<512xi16>, %bits: memref<32xi8>) {
  // expected-error @+1 {{is not an exact realization: 16-bit metrics, symbols within 127, unreachable seed -31244, renormalization period 118}}
  ondsp.viterbi_decode %symbols, %bits {constraint_length = 7 : i64,
      polynomials = array<i64: 121, 91>, metric_bits = 16 : i64, symbol_bound = 127 : i64,
      unreachable_metric = -31244 : i64, renormalization_period = 118 : i64}
      : memref<512xi16>, memref<32xi8>
  return
}

// -----

// A seed at -2 (K - 1) D ties a reachable candidate at stage K - 1.
func.func @seed_too_high(%symbols: memref<128xi16>, %bits: memref<8xi8>) {
  // expected-error @+1 {{is not an exact realization}}
  ondsp.viterbi_decode %symbols, %bits {constraint_length = 7 : i64,
      polynomials = array<i64: 121, 91>, metric_bits = 16 : i64, symbol_bound = 127 : i64,
      unreachable_metric = -3048 : i64, renormalization_period = 0 : i64}
      : memref<128xi16>, memref<8xi8>
  return
}

// -----

// Full-range symbols cannot even form a branch metric in 16 bits.
func.func @unbounded_narrow(%symbols: memref<128xi16>, %bits: memref<8xi8>) {
  // expected-error @+1 {{is not an exact realization}}
  ondsp.viterbi_decode %symbols, %bits {constraint_length = 7 : i64,
      polynomials = array<i64: 121, 91>, metric_bits = 16 : i64, symbol_bound = 32768 : i64,
      unreachable_metric = -32768 : i64, renormalization_period = 0 : i64}
      : memref<128xi16>, memref<8xi8>
  return
}

// -----

func.func @frame_not_bytes(%symbols: memref<24xi16>, %bits: memref<1xi8>) {
  // expected-error @+1 {{requires N * R symbols with N a positive multiple of 8}}
  ondsp.viterbi_decode %symbols, %bits {constraint_length = 3 : i64,
      polynomials = array<i64: 7, 5>, metric_bits = 32 : i64, symbol_bound = 32768 : i64,
      unreachable_metric = -1073741825 : i64, renormalization_period = 0 : i64}
      : memref<24xi16>, memref<1xi8>
  return
}

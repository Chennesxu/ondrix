// RUN: ondrix-opt %s --split-input-file --verify-diagnostics

func.func @constraint_too_long(%symbols: tensor<32xi16>) -> tensor<2xi8> {
  // expected-error @+1 {{constraint_length must be in [3, 7]}}
  %bits = ondrix.viterbi_decode %symbols {constraint_length = 8 : i64,
      polynomials = array<i64: 7, 5>} : (tensor<32xi16>) -> tensor<2xi8>
  return %bits : tensor<2xi8>
}

// -----

func.func @rate_one_quarter(%symbols: tensor<64xi16>) -> tensor<2xi8> {
  // expected-error @+1 {{requires two or three generator polynomials}}
  %bits = ondrix.viterbi_decode %symbols {constraint_length = 3 : i64,
      polynomials = array<i64: 7, 5, 6, 3>} : (tensor<64xi16>) -> tensor<2xi8>
  return %bits : tensor<2xi8>
}

// -----

func.func @generator_too_wide(%symbols: tensor<32xi16>) -> tensor<2xi8> {
  // expected-error @+1 {{generator polynomial 8 is not in [1, 2^3)}}
  %bits = ondrix.viterbi_decode %symbols {constraint_length = 3 : i64,
      polynomials = array<i64: 7, 8>} : (tensor<32xi16>) -> tensor<2xi8>
  return %bits : tensor<2xi8>
}

// -----

func.func @frame_not_bytes(%symbols: tensor<24xi16>) -> tensor<1xi8> {
  // expected-error @+1 {{requires N * R symbols with N a positive multiple of 8}}
  %bits = ondrix.viterbi_decode %symbols {constraint_length = 3 : i64,
      polynomials = array<i64: 7, 5>} : (tensor<24xi16>) -> tensor<1xi8>
  return %bits : tensor<1xi8>
}

// -----

// One frame past the bound the i32 metrics rest on.
func.func @frame_past_bound(%symbols: tensor<16400xi16>) -> tensor<1025xi8> {
  // expected-error @+1 {{N * R <= 16384}}
  %bits = ondrix.viterbi_decode %symbols {constraint_length = 3 : i64,
      polynomials = array<i64: 7, 5>} : (tensor<16400xi16>) -> tensor<1025xi8>
  return %bits : tensor<1025xi8>
}

// -----

func.func @bits_miscounted(%symbols: tensor<32xi16>) -> tensor<3xi8> {
  // expected-error @+1 {{packs 16 decoded bits into 2 bytes, not 3}}
  %bits = ondrix.viterbi_decode %symbols {constraint_length = 3 : i64,
      polynomials = array<i64: 7, 5>} : (tensor<32xi16>) -> tensor<3xi8>
  return %bits : tensor<3xi8>
}

// -----

func.func @symbols_not_i16(%symbols: tensor<32xi32>) -> tensor<2xi8> {
  // expected-error @+1 {{requires a static tensor<(N*R)xi16> of symbols}}
  %bits = ondrix.viterbi_decode %symbols {constraint_length = 3 : i64,
      polynomials = array<i64: 7, 5>} : (tensor<32xi32>) -> tensor<2xi8>
  return %bits : tensor<2xi8>
}

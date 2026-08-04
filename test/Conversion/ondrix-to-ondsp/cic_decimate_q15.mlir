// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --split-input-file | FileCheck %s

// The declared overflow reaches every state combine and only the export
// rounds, so the mode a source program picks is visible on each boundary.

// CHECK-LABEL: func.func @cic_wrap
// The carrier is 16 + stages * log2(rate * delay) = 20 bits wide.
// CHECK: arith.constant 0 : i20
// CHECK-COUNT-2: ondsp.add_shift {{.*}}overflow = wrap, saturate_to = i20
// CHECK-COUNT-2: ondsp.sub_shift {{.*}}overflow = wrap, saturate_to = i20
// CHECK: ondsp.round_shift {{.*}}post_shift_right = 4, rounding = nearest_even, overflow = saturate, saturate_to = i16
// CHECK-NOT: ondsp.add_shift
// CHECK-NOT: ondsp.sub_shift
func.func @cic_wrap(%input: tensor<32xi16>) -> tensor<8xi16> {
  %result = ondrix.cic_decimate %input {
    stages = 2 : i64,
    rate = 4 : i64,
    delay = 1 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<wrap>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<32xi16>) -> tensor<8xi16>
  return %result : tensor<8xi16>
}

// -----

// CHECK-LABEL: func.func @cic_saturate
// CHECK-COUNT-2: ondsp.add_shift {{.*}}overflow = saturate, saturate_to = i20
// CHECK-COUNT-2: ondsp.sub_shift {{.*}}overflow = saturate, saturate_to = i20
func.func @cic_saturate(%input: tensor<32xi16>) -> tensor<8xi16> {
  %result = ondrix.cic_decimate %input {
    stages = 2 : i64,
    rate = 4 : i64,
    delay = 1 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<32xi16>) -> tensor<8xi16>
  return %result : tensor<8xi16>
}

// -----

// A differential delay of two gives each comb stage a two-deep line, so the
// loop carries stages * delay state values plus the result tensor.
// CHECK-LABEL: func.func @cic_delay_two
// CHECK: arith.constant 0 : i28
// CHECK: scf.for {{.*}}iter_args({{.*}}) -> (i28, i28, i28, i28, i28, i28, tensor<8xi16>)
// CHECK: ondsp.round_shift {{.*}}post_shift_right = 12
func.func @cic_delay_two(%input: tensor<64xi16>) -> tensor<8xi16> {
  %result = ondrix.cic_decimate %input {
    stages = 3 : i64,
    rate = 8 : i64,
    delay = 2 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<wrap>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<64xi16>) -> tensor<8xi16>
  return %result : tensor<8xi16>
}

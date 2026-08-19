// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/cic_decimate_q15_aot.c %t.o -o %t
// RUN: %t

// The two rate-four kernels differ only in the declared state overflow. The
// harness asserts they diverge on a full-scale corpus, so the mode gate
// cannot pass by both programs coinciding.

func.func @cic_s1_r2_m1_wrap(%input: tensor<32xi16>) -> tensor<16xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.cic_decimate %input {
    stages = 1 : i64,
    rate = 2 : i64,
    delay = 1 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<wrap>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<32xi16>) -> tensor<16xi16>
  return %result : tensor<16xi16>
}

func.func @cic_s2_r4_m1_wrap(%input: tensor<32xi16>) -> tensor<8xi16>
    attributes {llvm.emit_c_interface} {
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

func.func @cic_s2_r4_m1_saturate(%input: tensor<32xi16>) -> tensor<8xi16>
    attributes {llvm.emit_c_interface} {
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

func.func @cic_s3_r8_m2_wrap(%input: tensor<64xi16>) -> tensor<8xi16>
    attributes {llvm.emit_c_interface} {
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

func.func @cic_s4_r16_m1_wrap(%input: tensor<64xi16>) -> tensor<4xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.cic_decimate %input {
    stages = 4 : i64,
    rate = 16 : i64,
    delay = 1 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<wrap>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<64xi16>) -> tensor<4xi16>
  return %result : tensor<4xi16>
}

// The W = 64 admission ceiling: growth 8 * log2(64) = 48 over i16 input.
func.func @cic_s8_r64_m1_wrap(%input: tensor<128xi16>) -> tensor<2xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.cic_decimate %input {
    stages = 8 : i64,
    rate = 64 : i64,
    delay = 1 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<wrap>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<128xi16>) -> tensor<2xi16>
  return %result : tensor<2xi16>
}

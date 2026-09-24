// RUN: ondrix-opt %s --ondrix-default-pipeline > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/viterbi_aot.c %t.o -o %t
// RUN: %t

// Three codes against an independent reference decoder, noiseless, noisy,
// full-scale and all-tie frames, and one frame at the N * R = 16384 bound.

func.func @viterbi_k7_r2(%symbols: tensor<512xi16>) -> tensor<32xi8>
    attributes {llvm.emit_c_interface} {
  %bits = ondrix.viterbi_decode %symbols {constraint_length = 7 : i64,
      polynomials = array<i64: 121, 91>} : (tensor<512xi16>) -> tensor<32xi8>
  return %bits : tensor<32xi8>
}

func.func @viterbi_k5_r3(%symbols: tensor<192xi16>) -> tensor<8xi8>
    attributes {llvm.emit_c_interface} {
  %bits = ondrix.viterbi_decode %symbols {constraint_length = 5 : i64,
      polynomials = array<i64: 21, 27, 31>} : (tensor<192xi16>) -> tensor<8xi8>
  return %bits : tensor<8xi8>
}

func.func @viterbi_k3_r2(%symbols: tensor<32xi16>) -> tensor<2xi8>
    attributes {llvm.emit_c_interface} {
  %bits = ondrix.viterbi_decode %symbols {constraint_length = 3 : i64,
      polynomials = array<i64: 7, 5>} : (tensor<32xi16>) -> tensor<2xi8>
  return %bits : tensor<2xi8>
}

func.func @viterbi_k7_bound(%symbols: tensor<16384xi16>) -> tensor<1024xi8>
    attributes {llvm.emit_c_interface} {
  %bits = ondrix.viterbi_decode %symbols {constraint_length = 7 : i64,
      polynomials = array<i64: 121, 91>} : (tensor<16384xi16>) -> tensor<1024xi8>
  return %bits : tensor<1024xi8>
}

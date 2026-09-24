// RUN: ondrix-opt %s --narrow-ondsp-viterbi-metrics --lower-ondsp-assumptions --convert-ondsp-to-ortumcore > %t.unit.mlir
// RUN: FileCheck %s < %t.unit.mlir
// RUN: ondrix-opt %t.unit.mlir --convert-ortumcore-to-ondsp-emulation --convert-ondsp-fixed-to-scalar --buffer-deallocation --expand-strided-metadata --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc -DTRELLIS_ONLY %S/Inputs/viterbi_narrow_aot.c %t.o -o %t
// RUN: %t

// The K = 7 decoders reach the trellis unit and decode, through its public
// emulation, what the contract's reference decodes.
// CHECK-LABEL: func.func @k7_short
// CHECK: ortumcore.viterbi_decode
// CHECK-LABEL: func.func @k7_long
// CHECK: ortumcore.viterbi_decode
// CHECK-LABEL: func.func @k7_rail
// CHECK: ortumcore.viterbi_decode

func.func @k7_short(%symbols: memref<128xi16>, %bits: memref<8xi8>)
    attributes {llvm.emit_c_interface} {
  %bounded = ondsp.assume_magnitude_bound %symbols {bound = 127 : i64} : memref<128xi16>
  ondsp.viterbi_decode %bounded, %bits {constraint_length = 7 : i64,
      polynomials = array<i64: 121, 91>, metric_bits = 32 : i64,
      symbol_bound = 32768 : i64, unreachable_metric = -1073741825 : i64,
      renormalization_period = 0 : i64} : memref<128xi16>, memref<8xi8>
  return
}

func.func @k7_long(%symbols: memref<1024xi16>, %bits: memref<64xi8>)
    attributes {llvm.emit_c_interface} {
  %bounded = ondsp.assume_magnitude_bound %symbols {bound = 127 : i64} : memref<1024xi16>
  ondsp.viterbi_decode %bounded, %bits {constraint_length = 7 : i64,
      polynomials = array<i64: 121, 91>, metric_bits = 32 : i64,
      symbol_bound = 32768 : i64, unreachable_metric = -1073741825 : i64,
      renormalization_period = 0 : i64} : memref<1024xi16>, memref<64xi8>
  return
}

func.func @k7_rail(%symbols: memref<1024xi16>, %bits: memref<64xi8>)
    attributes {llvm.emit_c_interface} {
  %bounded = ondsp.assume_magnitude_bound %symbols {bound = 900 : i64} : memref<1024xi16>
  ondsp.viterbi_decode %bounded, %bits {constraint_length = 7 : i64,
      polynomials = array<i64: 121, 91>, metric_bits = 32 : i64,
      symbol_bound = 32768 : i64, unreachable_metric = -1073741825 : i64,
      renormalization_period = 0 : i64} : memref<1024xi16>, memref<64xi8>
  return
}

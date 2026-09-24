// RUN: ondrix-opt %s --narrow-ondsp-viterbi-metrics --lower-ondsp-assumptions > %t.narrow.mlir
// RUN: FileCheck %s < %t.narrow.mlir
// RUN: ondrix-opt %t.narrow.mlir --convert-ondsp-fixed-to-scalar --buffer-deallocation --expand-strided-metadata --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/viterbi_narrow_aot.c %t.o -o %t
// RUN: %t

// Declared symbol bounds certify 16-bit metrics: no renormalization, every
// 117 stages, and every 6 (the tightest period the certificate admits).
// CHECK-LABEL: func.func @k7_short
// CHECK: metric_bits = 16 : i64{{.*}}renormalization_period = 0 : i64
// CHECK-LABEL: func.func @k7_long
// CHECK: metric_bits = 16 : i64{{.*}}renormalization_period = 117 : i64
// CHECK-LABEL: func.func @k7_rail
// CHECK: metric_bits = 16 : i64{{.*}}renormalization_period = 6 : i64
// CHECK-LABEL: func.func @k5_r3
// CHECK: metric_bits = 16 : i64{{.*}}renormalization_period = 46 : i64
// CHECK-LABEL: func.func @k3_r2
// CHECK: metric_bits = 16 : i64{{.*}}renormalization_period = 4 : i64

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

func.func @k5_r3(%symbols: memref<192xi16>, %bits: memref<8xi8>)
    attributes {llvm.emit_c_interface} {
  %bounded = ondsp.assume_magnitude_bound %symbols {bound = 200 : i64} : memref<192xi16>
  ondsp.viterbi_decode %bounded, %bits {constraint_length = 5 : i64,
      polynomials = array<i64: 21, 27, 31>, metric_bits = 32 : i64,
      symbol_bound = 32768 : i64, unreachable_metric = -1073741825 : i64,
      renormalization_period = 0 : i64} : memref<192xi16>, memref<8xi8>
  return
}

func.func @k3_r2(%symbols: memref<512xi16>, %bits: memref<32xi8>)
    attributes {llvm.emit_c_interface} {
  %bounded = ondsp.assume_magnitude_bound %symbols {bound = 2000 : i64} : memref<512xi16>
  ondsp.viterbi_decode %bounded, %bits {constraint_length = 3 : i64,
      polynomials = array<i64: 7, 5>, metric_bits = 32 : i64,
      symbol_bound = 32768 : i64, unreachable_metric = -1073741825 : i64,
      renormalization_period = 0 : i64} : memref<512xi16>, memref<32xi8>
  return
}

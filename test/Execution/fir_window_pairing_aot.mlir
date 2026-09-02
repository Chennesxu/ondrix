// RUN: ondrix-opt %s --convert-ondrix-to-ondsp=preserve-bufferizable-reductions=true --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" --canonicalize --widen-ondsp-exact-accumulators --pair-ondsp-fixed-reduction-outputs > %t.paired.mlir
// RUN: FileCheck %s --check-prefix=PAIRED < %t.paired.mlir
// RUN: ondrix-opt %t.paired.mlir --scalarize-ondsp-fixed-reduce-mac --convert-ondsp-fixed-to-scalar --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/fir_window_pairing_aot.c %t.o -o %t
// RUN: %t

// The window-loop pairing on the generic path: every shape it produces (a pair
// loop with an odd remainder, an unrolled pair walk with one, and an even
// count with none) executes bit-exactly against the ordered reference.

// PAIRED-LABEL: func.func @fir_pairs_loop_odd
// PAIRED: memref.alloca() {alignment = 4 : i64} : memref<66xi16>
// PAIRED: scf.for
// PAIRED: ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 2>
// PAIRED: ondsp.reduce_mac
func.func @fir_pairs_loop_odd(%input: tensor<64xi16>, %coeffs: tensor<32xi16>) -> tensor<33xi16>
    attributes {llvm.emit_c_interface} {
  %init = tensor.empty() : tensor<33xi16>
  %result = ondrix.fir_filter %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<valid>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_ties_positive>
  } : (tensor<64xi16>, tensor<32xi16>, tensor<33xi16>) -> tensor<33xi16>
  return %result : tensor<33xi16>
}

// PAIRED-LABEL: func.func @fir_pairs_unrolled_odd
// PAIRED: memref.alloca() {alignment = 4 : i64} : memref<10xi16>
// PAIRED-COUNT-4: ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 2>
// PAIRED: ondsp.reduce_mac
func.func @fir_pairs_unrolled_odd(%input: tensor<12xi16>, %coeffs: tensor<4xi16>) -> tensor<9xi16>
    attributes {llvm.emit_c_interface} {
  %init = tensor.empty() : tensor<9xi16>
  %result = ondrix.fir_filter %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<valid>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_ties_positive>
  } : (tensor<12xi16>, tensor<4xi16>, tensor<9xi16>) -> tensor<9xi16>
  return %result : tensor<9xi16>
}

// PAIRED-LABEL: func.func @fir_pairs_even
// PAIRED-COUNT-5: ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 2>
// PAIRED-NOT: ondsp.reduce_mac
func.func @fir_pairs_even(%input: tensor<13xi16>, %coeffs: tensor<4xi16>) -> tensor<10xi16>
    attributes {llvm.emit_c_interface} {
  %init = tensor.empty() : tensor<10xi16>
  %result = ondrix.fir_filter %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<valid>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_ties_positive>
  } : (tensor<13xi16>, tensor<4xi16>, tensor<10xi16>) -> tensor<10xi16>
  return %result : tensor<10xi16>
}

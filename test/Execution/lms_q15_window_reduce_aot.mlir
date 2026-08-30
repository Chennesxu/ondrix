// RUN: ondrix-opt %s --ondrix-default-pipeline=vector-bits=256 > %t.mlir
// RUN: FileCheck %s --check-prefix=BATCHED --implicit-check-not=ondsp. < %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/lms_q15_window_reduce_aot.c %t.o -o %t
// RUN: %t

// The canonical pipeline, so the tap reduction and the weight update are both
// batched and each one's lane order has to be right for the harness to pass.

// BATCHED-LABEL: llvm.func @lms_q15_window
// Eleven taps at width eight: one folded block, three terms left ordered.
// BATCHED: llvm.shufflevector %{{.*}} [7, 6, 5, 4, 3, 2, 1, 0] : vector<8xi16>
// BATCHED: llvm.intr.vector.reduce.add

func.func @lms_q15_window(%x: tensor<64xi16>, %d: tensor<64xi16>, %w: tensor<11xi16>)
    -> (tensor<64xi16>, tensor<11xi16>) attributes {llvm.emit_c_interface} {
  %error, %adapted = ondrix.lms %x, %d, %w {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>,
    step_size = 4096 : i64
  } : (tensor<64xi16>, tensor<64xi16>, tensor<11xi16>) -> (tensor<64xi16>, tensor<11xi16>)
  return %error, %adapted : tensor<64xi16>, tensor<11xi16>
}

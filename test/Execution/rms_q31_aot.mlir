// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/rms_q31_aot.c %t.o -o %t -lm
// RUN: %t
// The canonical pipeline is a SEPARATE route: it preserves bufferizable
// reductions, so a profile the reduce_mac vocabulary cannot carry has to fall
// back here rather than fail. Compiling through it is the only thing that
// checks that.
// RUN: ondrix-opt %s --ondrix-default-pipeline="vector-bits=256" > %t.pipeline.mlir
// RUN: ondrix-translate %t.pipeline.mlir --mlir-to-llvmir > %t.pipeline.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.pipeline.ll -o %t.pipeline.o
// RUN: cc %S/Inputs/rms_q31_aot.c %t.pipeline.o -o %t.pipeline -lm
// RUN: %t.pipeline

// The Q31 profile against an independent reference. Two extents carry two
// DIFFERENT derived pre-shifts (N=16 -> k=2, N=64 -> k=3), so a reference
// that hardcoded one of them fails the other.

func.func @rms16_q31(%input: tensor<16xi32>) -> tensor<1xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.rms %input {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    input_rounding = #ondsp.rounding<nearest_even>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<16xi32>) -> tensor<1xi32>
  return %result : tensor<1xi32>
}

func.func @rms64_q31(%input: tensor<64xi32>) -> tensor<1xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.rms %input {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    input_rounding = #ondsp.rounding<nearest_even>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<64xi32>) -> tensor<1xi32>
  return %result : tensor<1xi32>
}

func.func @rms64_q31_floor(%input: tensor<64xi32>) -> tensor<1xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.rms %input {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    input_rounding = #ondsp.rounding<toward_negative>,
    rounding = #ondsp.rounding<toward_negative>
  } : (tensor<64xi32>) -> tensor<1xi32>
  return %result : tensor<1xi32>
}

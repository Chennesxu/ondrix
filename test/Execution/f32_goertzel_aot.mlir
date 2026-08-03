// RUN: ondrix-opt %s --ondrix-default-pipeline > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc -ffp-contract=off %S/Inputs/f32_goertzel_aot.c %t.o -lm -o %t
// RUN: %t

// off and fma are exact contracts, so those are bit for bit against a
// reference that runs the declared event graph itself. fast is checked for
// membership in the two-element derivable set instead: the flagged event is
// free to be fused or not, and on a target without an FMA instruction the
// backend expands it.

func.func @f32_goertzel_off(%input: tensor<16xf32>) -> tensor<1xf32>
    attributes {llvm.emit_c_interface} {
  %energy = ondrix.goertzel %input {
    bin = 3,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<16xf32>) -> tensor<1xf32>
  return %energy : tensor<1xf32>
}

func.func @f32_goertzel_fma(%input: tensor<16xf32>) -> tensor<1xf32>
    attributes {llvm.emit_c_interface} {
  %energy = ondrix.goertzel %input {
    bin = 3,
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<16xf32>) -> tensor<1xf32>
  return %energy : tensor<1xf32>
}

func.func @f32_goertzel_fast(%input: tensor<16xf32>) -> tensor<1xf32>
    attributes {llvm.emit_c_interface} {
  %energy = ondrix.goertzel %input {
    bin = 3,
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (tensor<16xf32>) -> tensor<1xf32>
  return %energy : tensor<1xf32>
}

func.func @f32_goertzel_quarter_turn(%input: tensor<16xf32>) -> tensor<1xf32>
    attributes {llvm.emit_c_interface} {
  %energy = ondrix.goertzel %input {
    bin = 4,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<16xf32>) -> tensor<1xf32>
  return %energy : tensor<1xf32>
}

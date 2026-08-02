// RUN: ondrix-compile --emit=contracts %S/../Frontend/Inputs/f32_matmul.ox | ondrix-opt --ondrix-default-pipeline > %t.matmul.mlir
// RUN: ondrix-translate %t.matmul.mlir --mlir-to-llvmir > %t.matmul.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.matmul.ll -o %t.matmul.o
// RUN: ondrix-opt %s --ondrix-default-pipeline > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc -ffp-contract=off %S/Inputs/f32_matmul_rms_aot.c %t.o %t.matmul.o -lm -o %t
// RUN: %t

// Both f32 profiles are exact contracts, so this gate is bit-for-bit against
// an independent reference: matmul under off and fma, and rms under both, at
// an extent that is not a power of two. The .ox object contributes the
// source-level binding of the same matmul contract.

func.func @f32_matmul_off(%a: tensor<3x4xf32>, %b: tensor<4x2xf32>) -> tensor<3x2xf32>
    attributes {llvm.emit_c_interface} {
  %c = ondrix.matmul %a, %b {
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<3x4xf32>, tensor<4x2xf32>) -> tensor<3x2xf32>
  return %c : tensor<3x2xf32>
}

func.func @f32_matmul_fma(%a: tensor<3x4xf32>, %b: tensor<4x2xf32>) -> tensor<3x2xf32>
    attributes {llvm.emit_c_interface} {
  %c = ondrix.matmul %a, %b {
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<3x4xf32>, tensor<4x2xf32>) -> tensor<3x2xf32>
  return %c : tensor<3x2xf32>
}

func.func @f32_rms_off(%input: tensor<10xf32>) -> tensor<1xf32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.rms %input {
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<10xf32>) -> tensor<1xf32>
  return %result : tensor<1xf32>
}

func.func @f32_rms_fma(%input: tensor<10xf32>) -> tensor<1xf32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.rms %input {
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<10xf32>) -> tensor<1xf32>
  return %result : tensor<1xf32>
}

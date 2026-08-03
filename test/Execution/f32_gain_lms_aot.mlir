// RUN: ondrix-compile --emit=contracts %S/../Frontend/Inputs/f32_lms.ox | ondrix-opt --ondrix-default-pipeline > %t.lms.mlir
// RUN: ondrix-translate %t.lms.mlir --mlir-to-llvmir > %t.lms.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.lms.ll -o %t.lms.o
// RUN: ondrix-opt %s --ondrix-default-pipeline > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc -ffp-contract=off %S/Inputs/f32_gain_lms_aot.c %t.o %t.lms.o -lm -o %t
// RUN: %t

// Requires gain's three objects to agree bit for bit with each other and
// with the reference, and pins lms against a per-step reference where one
// differing weight compounds through every later sample.

func.func @f32_gain_off(%input: tensor<16xf32>) -> tensor<16xf32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.gain %input {
    fp_gain = 3.750000e-01 : f32,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<16xf32>) -> tensor<16xf32>
  return %result : tensor<16xf32>
}

func.func @f32_gain_fma(%input: tensor<16xf32>) -> tensor<16xf32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.gain %input {
    fp_gain = 3.750000e-01 : f32,
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<16xf32>) -> tensor<16xf32>
  return %result : tensor<16xf32>
}

func.func @f32_gain_fast(%input: tensor<16xf32>) -> tensor<16xf32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.gain %input {
    fp_gain = 3.750000e-01 : f32,
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (tensor<16xf32>) -> tensor<16xf32>
  return %result : tensor<16xf32>
}

func.func @f32_lms_off(%input: tensor<32xf32>, %desired: tensor<32xf32>, %weights: tensor<4xf32>)
    -> (tensor<32xf32>, tensor<4xf32>) attributes {llvm.emit_c_interface} {
  %error, %adapted = ondrix.lms %input, %desired, %weights {
    fp_step_size = 6.250000e-02 : f32,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<32xf32>, tensor<32xf32>, tensor<4xf32>) -> (tensor<32xf32>, tensor<4xf32>)
  return %error, %adapted : tensor<32xf32>, tensor<4xf32>
}

func.func @f32_lms_fma(%input: tensor<32xf32>, %desired: tensor<32xf32>, %weights: tensor<4xf32>)
    -> (tensor<32xf32>, tensor<4xf32>) attributes {llvm.emit_c_interface} {
  %error, %adapted = ondrix.lms %input, %desired, %weights {
    fp_step_size = 6.250000e-02 : f32,
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<32xf32>, tensor<32xf32>, tensor<4xf32>) -> (tensor<32xf32>, tensor<4xf32>)
  return %error, %adapted : tensor<32xf32>, tensor<4xf32>
}

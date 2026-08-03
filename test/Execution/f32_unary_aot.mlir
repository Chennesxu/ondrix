// RUN: ondrix-opt %s --ondrix-default-pipeline > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc -ffp-contract=off %S/Inputs/f32_unary_aot.c %t.o -lm -o %t
// RUN: %t

// Exact contracts, so both profiles are gated bit for bit. The averages are
// recomputed per window under every contract, which is what makes the off
// and fma legs of one window separable at all.

func.func @f32_moving_average_off(%input: tensor<8xf32>) -> tensor<6xf32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.moving_average %input {
    window = 3 : i64, numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<8xf32>) -> tensor<6xf32>
  return %result : tensor<6xf32>
}

func.func @f32_dct_off(%input: tensor<8xf32>) -> tensor<8xf32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fp<format = f32, contract = off>,
    output_numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %result : tensor<8xf32>
}

func.func @f32_dct_fma(%input: tensor<8xf32>) -> tensor<8xf32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fp<format = f32, contract = fma>,
    output_numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %result : tensor<8xf32>
}

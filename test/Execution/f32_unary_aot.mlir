// RUN: ondrix-opt %s --ondrix-default-pipeline="vector-bits=256" > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc -ffp-contract=off %S/Inputs/f32_unary_aot.c %t.o -lm -o %t
// RUN: %t

// Exact contracts, so both profiles are gated bit for bit. The averages are
// recomputed per window under every contract, which is what makes the off
// and fma legs of one window separable at all.
//
// The three fast legs differ in what they spend. A moving average has no
// product to fuse; its window sum is a reduction tree, so R applies, but at
// K = 3 every chained tree is the declared left fold, so the rebuild refuses
// and the declared association is what runs (checkWindowAssociation gates
// that). At K = 8 the trees separate and the rebuild fires, so that leg runs
// the balanced tree its own reference recomputes. The DCT rows select the
// fused chain, which spends F.
//
// Both pin a SELECTION rather than the contract: fast may legally produce any
// member, so a transform that starts choosing differently must redden these
// and re-justify itself rather than change the object silently.

// One window over the whole input, so no output batching can reach it: the
// balanced tree here is the scalar lowering's own rebuild.
func.func @f32_moving_average_fast8(%input: tensor<8xf32>) -> tensor<1xf32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.moving_average %input {
    window = 8 : i64, numeric = #ondsp.fp<format = f32, contract = fast>
  } : (tensor<8xf32>) -> tensor<1xf32>
  return %result : tensor<1xf32>
}

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

func.func @f32_moving_average_fma(%input: tensor<8xf32>) -> tensor<6xf32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.moving_average %input {
    window = 3 : i64, numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<8xf32>) -> tensor<6xf32>
  return %result : tensor<6xf32>
}

func.func @f32_moving_average_fast(%input: tensor<8xf32>) -> tensor<6xf32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.moving_average %input {
    window = 3 : i64, numeric = #ondsp.fp<format = f32, contract = fast>
  } : (tensor<8xf32>) -> tensor<6xf32>
  return %result : tensor<6xf32>
}

func.func @f32_dct_fast(%input: tensor<8xf32>) -> tensor<8xf32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fp<format = f32, contract = fast>,
    output_numeric = #ondsp.fp<format = f32, contract = fast>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %result : tensor<8xf32>
}

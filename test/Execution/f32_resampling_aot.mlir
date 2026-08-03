// RUN: ondrix-compile --emit=contracts %S/../Frontend/Inputs/f32_fir_decimate.ox | ondrix-opt --ondrix-default-pipeline="vector-bits=256" > %t.decimate.mlir
// RUN: ondrix-translate %t.decimate.mlir --mlir-to-llvmir > %t.decimate.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.decimate.ll -o %t.decimate.o
// RUN: ondrix-compile --emit=contracts %S/../Frontend/Inputs/f32_fir_interpolate.ox | ondrix-opt --ondrix-default-pipeline="vector-bits=256" > %t.interpolate.mlir
// RUN: ondrix-translate %t.interpolate.mlir --mlir-to-llvmir > %t.interpolate.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.interpolate.ll -o %t.interpolate.o
// RUN: ondrix-opt %s --ondrix-default-pipeline="vector-bits=256" > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc -ffp-contract=off %S/Inputs/f32_resampling_aot.c %t.o %t.decimate.o %t.interpolate.o -lm -o %t
// RUN: %t

// Both f32 resampling profiles are exact contracts, so this gate is bit for
// bit against a reference that walks the declared index relation itself. The
// two source-level objects contribute the same contracts through the .ox
// binding.

func.func @f32_decimate_off(%input: tensor<12xf32>, %coeffs: tensor<5xf32>) -> tensor<4xf32>
    attributes {llvm.emit_c_interface} {
  %init = tensor.empty() : tensor<4xf32>
  %result = ondrix.fir_decimate %input, %coeffs, %init {
    factor = 2,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<12xf32>, tensor<5xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %result : tensor<4xf32>
}

func.func @f32_decimate_fma(%input: tensor<12xf32>, %coeffs: tensor<5xf32>) -> tensor<4xf32>
    attributes {llvm.emit_c_interface} {
  %init = tensor.empty() : tensor<4xf32>
  %result = ondrix.fir_decimate %input, %coeffs, %init {
    factor = 2,
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<12xf32>, tensor<5xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %result : tensor<4xf32>
}

func.func @f32_interpolate_off(%input: tensor<4xf32>, %coeffs: tensor<3xf32>) -> tensor<9xf32>
    attributes {llvm.emit_c_interface} {
  %init = tensor.empty() : tensor<9xf32>
  %result = ondrix.fir_interpolate %input, %coeffs, %init {
    factor = 2,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<4xf32>, tensor<3xf32>, tensor<9xf32>) -> tensor<9xf32>
  return %result : tensor<9xf32>
}

func.func @f32_interpolate_fma(%input: tensor<4xf32>, %coeffs: tensor<3xf32>) -> tensor<9xf32>
    attributes {llvm.emit_c_interface} {
  %init = tensor.empty() : tensor<9xf32>
  %result = ondrix.fir_interpolate %input, %coeffs, %init {
    factor = 2,
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<4xf32>, tensor<3xf32>, tensor<9xf32>) -> tensor<9xf32>
  return %result : tensor<9xf32>
}

// Interpolation has no bufferization interface, so its reduction never reaches
// the horizontal rewrite and fast consumes nothing. The selected member is the
// fused one; pinning it keeps a later transform from changing the object here
// without re-justifying itself.
func.func @f32_interpolate_fast(%input: tensor<4xf32>, %coeffs: tensor<3xf32>) -> tensor<9xf32>
    attributes {llvm.emit_c_interface} {
  %init = tensor.empty() : tensor<9xf32>
  %result = ondrix.fir_interpolate %input, %coeffs, %init {
    factor = 2 : i64,
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (tensor<4xf32>, tensor<3xf32>, tensor<9xf32>) -> tensor<9xf32>
  return %result : tensor<9xf32>
}

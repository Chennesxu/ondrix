// RUN: ondrix-opt %s --ondrix-default-pipeline="vector-bits=256" > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: ondrix-compile --emit=llvm %S/../Frontend/Inputs/f32_phase_spectrum.ox | ondrix-translate --mlir-to-llvmir > %t.ox.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ox.ll -o %t.ox.o
// RUN: cc -ffp-contract=off %S/Inputs/cx_phase_f32_aot.c %t.o %t.ox.o -lm -o %t
// RUN: %t

// Three claims, each gated separately: the emitted turn is bit for bit the
// declared event graph, it stays inside two binary32 steps of the exact
// argument, and it is EXACT on the four axes and four diagonals.

func.func @cx_phase_off(%input: tensor<2048xf32>) -> tensor<1024xf32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.cx_phase %input {
    layout = #ondsp.cx_layout<interleaved>,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<2048xf32>) -> tensor<1024xf32>
  return %result : tensor<1024xf32>
}

func.func @cx_phase_fma(%input: tensor<2048xf32>) -> tensor<1024xf32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.cx_phase %input {
    layout = #ondsp.cx_layout<interleaved>,
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<2048xf32>) -> tensor<1024xf32>
  return %result : tensor<1024xf32>
}

func.func @f32_rfft64_off(%input: tensor<64xf32>) -> tensor<66xf32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.rfft %input {
    layout = #ondsp.cx_layout<interleaved>,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<64xf32>) -> tensor<66xf32>
  return %result : tensor<66xf32>
}

func.func @cx_phase64_off(%input: tensor<66xf32>) -> tensor<33xf32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.cx_phase %input {
    layout = #ondsp.cx_layout<interleaved>,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<66xf32>) -> tensor<33xf32>
  return %result : tensor<33xf32>
}

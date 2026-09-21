// RUN: ondrix-opt %s --ondrix-default-pipeline="vector-bits=256" > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: ondrix-compile --emit=llvm %S/../Frontend/Inputs/f32_cx_fir.ox | ondrix-translate --mlir-to-llvmir > %t.ox.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ox.ll -o %t.ox.o
// RUN: cc -ffp-contract=off %S/Inputs/cx_fir_filter_f32_aot.c %t.o %t.ox.o -lm -o %t
// RUN: %t

// The sliding form is the scalar reduction under a window, so this gate
// checks that the window is the only thing added: each output must equal the
// reduction of its own window, bit for bit, under both contract modes.

func.func @cx_fir_f32_off(%signal: tensor<24xf32>, %taps: tensor<8xf32>) -> tensor<18xf32>
    attributes {llvm.emit_c_interface} {
  %init = tensor.empty() : tensor<18xf32>
  %r = ondrix.cx_fir_filter %signal, %taps, %init {
    numeric = #ondsp.fp<format = f32, contract = off>,
    layout = #ondsp.cx_layout<interleaved>
  } : (tensor<24xf32>, tensor<8xf32>, tensor<18xf32>) -> tensor<18xf32>
  return %r : tensor<18xf32>
}

func.func @cx_fir_f32_conj_fma(%signal: tensor<24xf32>, %taps: tensor<8xf32>) -> tensor<18xf32>
    attributes {llvm.emit_c_interface} {
  %init = tensor.empty() : tensor<18xf32>
  %r = ondrix.cx_fir_filter %signal, %taps, %init {
    numeric = #ondsp.fp<format = f32, contract = fma>,
    layout = #ondsp.cx_layout<interleaved>,
    conjugate
  } : (tensor<24xf32>, tensor<8xf32>, tensor<18xf32>) -> tensor<18xf32>
  return %r : tensor<18xf32>
}

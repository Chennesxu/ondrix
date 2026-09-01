// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --ondrix-default-pipeline="vector-bits=256" > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: ondrix-opt %s --ondrix-default-pipeline="vector-bits=256" > %t.canonical.mlir
// RUN: ondrix-translate %t.canonical.mlir --mlir-to-llvmir > %t.canonical.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.canonical.ll -o %t.canonical.o
// RUN: ondrix-opt %s --convert-ondrix-to-ondsp="fft-loops" --ondrix-default-pipeline="vector-bits=256" > %t.loops.mlir
// RUN: ondrix-translate %t.loops.mlir --mlir-to-llvmir > %t.loops.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.loops.ll -o %t.loops.o
// RUN: ondrix-compile --emit=contracts %S/../Frontend/Inputs/f32_cfft_round_trip.ox | ondrix-opt --ondrix-default-pipeline="vector-bits=256" > %t.ox.mlir
// RUN: ondrix-translate %t.ox.mlir --mlir-to-llvmir > %t.ox.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ox.ll -o %t.ox.o
// RUN: cc -ffp-contract=off %S/Inputs/fft_f32_aot.c %t.o %t.ox.o -lm -o %t
// RUN: %t
// RUN: cc -ffp-contract=off %S/Inputs/fft_f32_aot.c %t.canonical.o %t.ox.o -lm -o %t.canonical
// RUN: %t.canonical
// RUN: cc -ffp-contract=off %S/Inputs/fft_f32_aot.c %t.loops.o %t.ox.o -lm -o %t.loops
// RUN: %t.loops

// The interleaved f32 transform has no requantization boundary, so this gate
// is bit for bit against a reference that walks the same event graph.

func.func @cfft8_off(%input: tensor<16xf32>) -> tensor<16xf32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<forward>,
    layout = #ondsp.cx_layout<interleaved>,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<16xf32>) -> tensor<16xf32>
  return %result : tensor<16xf32>
}

func.func @cfft8_fma(%input: tensor<16xf32>) -> tensor<16xf32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<forward>,
    layout = #ondsp.cx_layout<interleaved>,
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<16xf32>) -> tensor<16xf32>
  return %result : tensor<16xf32>
}

func.func @icfft8_off(%input: tensor<16xf32>) -> tensor<16xf32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<inverse>,
    layout = #ondsp.cx_layout<interleaved>,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<16xf32>) -> tensor<16xf32>
  return %result : tensor<16xf32>
}

func.func @cfft64_off(%input: tensor<128xf32>) -> tensor<128xf32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<forward>,
    layout = #ondsp.cx_layout<interleaved>,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<128xf32>) -> tensor<128xf32>
  return %result : tensor<128xf32>
}

func.func @cfft64_fma(%input: tensor<128xf32>) -> tensor<128xf32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<forward>,
    layout = #ondsp.cx_layout<interleaved>,
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<128xf32>) -> tensor<128xf32>
  return %result : tensor<128xf32>
}

func.func @rfft16_off(%input: tensor<16xf32>) -> tensor<18xf32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.rfft %input {
    layout = #ondsp.cx_layout<interleaved>,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<16xf32>) -> tensor<18xf32>
  return %result : tensor<18xf32>
}

func.func @irfft16_off(%input: tensor<18xf32>) -> tensor<16xf32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.irfft %input {
    layout = #ondsp.cx_layout<interleaved>,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<18xf32>) -> tensor<16xf32>
  return %result : tensor<16xf32>
}

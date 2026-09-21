// RUN: ondrix-opt %s --ondrix-default-pipeline="vector-bits=256" > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: ondrix-compile --emit=llvm %S/../Frontend/Inputs/f32_power_spectrum.ox | ondrix-translate --mlir-to-llvmir > %t.power.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.power.ll -o %t.power.o
// RUN: ondrix-compile --emit=llvm %S/../Frontend/Inputs/f32_magnitude_spectrum.ox | ondrix-translate --mlir-to-llvmir > %t.magnitude.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.magnitude.ll -o %t.magnitude.o
// RUN: cc -ffp-contract=off %S/Inputs/cx_spectrum_f32_aot.c %t.o %t.power.o %t.magnitude.o -lm -o %t
// RUN: %t

// The interleaved f32 readouts carry no requantization, so this gate is bit
// for bit against a reference that walks the same event graph.

func.func @cx_power_off(%input: tensor<16xf32>) -> tensor<8xf32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.cx_power %input {
    layout = #ondsp.cx_layout<interleaved>,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<16xf32>) -> tensor<8xf32>
  return %result : tensor<8xf32>
}

func.func @cx_power_fma(%input: tensor<16xf32>) -> tensor<8xf32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.cx_power %input {
    layout = #ondsp.cx_layout<interleaved>,
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<16xf32>) -> tensor<8xf32>
  return %result : tensor<8xf32>
}

func.func @cx_magnitude_off(%input: tensor<16xf32>) -> tensor<8xf32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.cx_magnitude %input {
    layout = #ondsp.cx_layout<interleaved>,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<16xf32>) -> tensor<8xf32>
  return %result : tensor<8xf32>
}

func.func @cx_magnitude_fma(%input: tensor<16xf32>) -> tensor<8xf32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.cx_magnitude %input {
    layout = #ondsp.cx_layout<interleaved>,
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<16xf32>) -> tensor<8xf32>
  return %result : tensor<8xf32>
}

// The two composed legs below exist so the source spellings are checked
// against the staged transform, not against a second FFT reference.

func.func @f32_rfft64_off(%input: tensor<64xf32>) -> tensor<66xf32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.rfft %input {
    layout = #ondsp.cx_layout<interleaved>,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<64xf32>) -> tensor<66xf32>
  return %result : tensor<66xf32>
}

func.func @cx_power64_off(%input: tensor<66xf32>) -> tensor<33xf32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.cx_power %input {
    layout = #ondsp.cx_layout<interleaved>,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<66xf32>) -> tensor<33xf32>
  return %result : tensor<33xf32>
}

func.func @f32_rfft64_fma(%input: tensor<64xf32>) -> tensor<66xf32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.rfft %input {
    layout = #ondsp.cx_layout<interleaved>,
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<64xf32>) -> tensor<66xf32>
  return %result : tensor<66xf32>
}

func.func @cx_magnitude64_fma(%input: tensor<66xf32>) -> tensor<33xf32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.cx_magnitude %input {
    layout = #ondsp.cx_layout<interleaved>,
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<66xf32>) -> tensor<33xf32>
  return %result : tensor<33xf32>
}

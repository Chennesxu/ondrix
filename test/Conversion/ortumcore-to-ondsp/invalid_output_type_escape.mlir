// RUN: not ondrix-opt %s --convert-ortumcore-to-ondsp-emulation 2>&1 | FileCheck %s

func.func @output_type_escape() -> !ortumcore.acc {
  // CHECK: error: failed to legalize operation 'ondsp.fft_stage'
  %result = ondsp.fft_stage {
    stage = 0 : i64,
    layout = #ondsp.cx_layout<split>,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : () -> !ortumcore.acc
  return %result : !ortumcore.acc
}

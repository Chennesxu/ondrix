// RUN: not ondrix-opt %s --convert-ortumcore-to-ondsp-emulation 2>&1 | FileCheck %s

func.func @nested(%value: tuple<!ortumcore.acc>) {
  // CHECK: error: 'func.func' op nested OrtumCore accumulator containers are unsupported
  return
}

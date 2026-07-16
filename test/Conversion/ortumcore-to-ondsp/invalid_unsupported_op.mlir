// RUN: not ondrix-opt %s --convert-ortumcore-to-ondsp-emulation 2>&1 | FileCheck %s

func.func @unsupported() {
  // CHECK: error: failed to legalize operation 'ortumcore.vec_state_init'
  %state = ortumcore.vec_state_init : !ortumcore.vec.state
  return
}

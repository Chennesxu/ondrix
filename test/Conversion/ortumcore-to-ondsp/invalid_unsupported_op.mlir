// RUN: not ondrix-opt %s --convert-ortumcore-to-ondsp-emulation 2>&1 | FileCheck %s

func.func @unsupported() {
  // CHECK: 'ortumcore.vec_state_init' op unsupported or nested OrtumCore type
  %state = ortumcore.vec_state_init : !ortumcore.vec.state
  return
}

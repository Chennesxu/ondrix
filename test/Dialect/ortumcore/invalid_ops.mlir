// RUN: not ondrix-opt %s 2>&1 | FileCheck %s

func.func @missing_fft_stage_kind(%s0: !ortumcore.vec.state, %a: i32) -> i32 {
  // CHECK: 'ortumcore.fft_trivial_stage' op requires attribute 'stage_kind'
  %s1, %o0 = ortumcore.fft_trivial_stage %s0, %a : (!ortumcore.vec.state, i32) -> (!ortumcore.vec.state, i32)
  return %o0 : i32
}

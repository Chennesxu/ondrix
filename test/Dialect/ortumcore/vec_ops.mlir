// RUN: ondrix-opt %s | FileCheck %s

// CHECK-LABEL: func.func @vec_ops
func.func @vec_ops(%a: i32, %b: i32) -> (i32, i32) {
  // CHECK: !ortumcore.vec.state
  %s0 = ortumcore.vec_state_init : !ortumcore.vec.state
  %s1 = ortumcore.vec_set_mode %s0 {sat = true, rnd = false, pack = true, shiftr = 15 : i64, shiftl = 0 : i64} : (!ortumcore.vec.state) -> !ortumcore.vec.state

  // CHECK: ortumcore.fft_trivial_stage
  // CHECK-SAME: stage_kind = #ortumcore<fft_stage_kind radix2>
  %s2, %o0, %o1 = ortumcore.fft_trivial_stage %s1, %a, %b {stage_kind = #ortumcore<fft_stage_kind radix2>} : (!ortumcore.vec.state, i32, i32) -> (!ortumcore.vec.state, i32, i32)
  // CHECK: ortumcore.fft_trivial_stage
  // CHECK-SAME: stage_kind = #ortumcore<fft_stage_kind radix2>
  %s3, %o2 = ortumcore.fft_trivial_stage %s2, %o0, %o1 {stage_kind = #ortumcore<fft_stage_kind radix2>} : (!ortumcore.vec.state, i32, i32) -> (!ortumcore.vec.state, i32)
  // CHECK: ortumcore.fft_trivial_stage
  // CHECK-SAME: stage_kind = #ortumcore<fft_stage_kind radix2_combined>
  %s4, %o3, %o4, %o5, %o6 = ortumcore.fft_trivial_stage %s3, %o1, %o2, %a, %b {stage_kind = #ortumcore<fft_stage_kind radix2_combined>} : (!ortumcore.vec.state, i32, i32, i32, i32) -> (!ortumcore.vec.state, i32, i32, i32, i32)
  return %o3, %o4 : i32, i32
}

// RUN: ondrix-opt %s | FileCheck %s

// CHECK-LABEL: func.func @vcu_ops
func.func @vcu_ops(%a: i32, %b: i32) -> (i32, i32) {
  // CHECK: !ortumcore.vec.state
  %s0 = ortumcore.vec_state_init : !ortumcore.vec.state
  %s1 = ortumcore.vec_set_mode %s0 {sat = true, rnd = false, cpack = true, shiftr = 15 : i64, shiftl = 0 : i64} : (!ortumcore.vec.state) -> !ortumcore.vec.state

  // CHECK: ortumcore.fft_primitive_7
  %s2, %o0, %o1 = ortumcore.fft_primitive_7 %s1, %a, %b : (!ortumcore.vec.state, i32, i32) -> (!ortumcore.vec.state, i32, i32)
  %s3, %o2 = ortumcore.fft_primitive_1 %s2, %o0, %o1 : (!ortumcore.vec.state, i32, i32) -> (!ortumcore.vec.state, i32)
  %s4, %o3, %o4, %o5, %o6 = ortumcore.fft_primitive_10 %s3, %o1, %o2, %a, %b : (!ortumcore.vec.state, i32, i32, i32, i32) -> (!ortumcore.vec.state, i32, i32, i32, i32)
  return %o3, %o4 : i32, i32
}

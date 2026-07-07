// RUN: not ondrix-opt %s 2>&1 | FileCheck %s

func.func @bad_fft_variant(%s0: !ortumcore.vec.state, %a: i32) -> i32 {
  // CHECK: 'ortumcore.fft_trivial_stage' op requires variant in range [1, 10]
  %s1, %o0 = ortumcore.fft_trivial_stage %s0, %a {variant = 999 : i64} : (!ortumcore.vec.state, i32) -> (!ortumcore.vec.state, i32)
  return %o0 : i32
}

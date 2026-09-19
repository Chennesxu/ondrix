// RUN: ondrix-opt %s --convert-ondsp-to-ortumcore | FileCheck %s

// The 32-bit complex accumulator carries as raw i32: its readout is a plain
// move, so it never enters the 40-bit accumulator domain.
func.func @cx_dot_to_ortumcore(
    %real: !ondsp.acc<storage = i32, frac = 30, signed, update_overflow = saturate>,
    %imag: !ondsp.acc<storage = i32, frac = 30, signed, update_overflow = saturate>,
    %lhs: memref<64xi32>, %rhs: memref<64xi32>)
    -> (!ondsp.acc<storage = i32, frac = 30, signed, update_overflow = saturate>,
        !ondsp.acc<storage = i32, frac = 30, signed, update_overflow = saturate>) {
  %re, %im = ondsp.cx_reduce_mac %real, %imag, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>
  } : (!ondsp.acc<storage = i32, frac = 30, signed, update_overflow = saturate>,
       !ondsp.acc<storage = i32, frac = 30, signed, update_overflow = saturate>,
       memref<64xi32>, memref<64xi32>)
      -> (!ondsp.acc<storage = i32, frac = 30, signed, update_overflow = saturate>,
          !ondsp.acc<storage = i32, frac = 30, signed, update_overflow = saturate>)
  return %re, %im : !ondsp.acc<storage = i32, frac = 30, signed, update_overflow = saturate>,
                    !ondsp.acc<storage = i32, frac = 30, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @cx_dot_to_ortumcore
// CHECK-SAME: (%{{.*}}: i32, %{{.*}}: i32, %{{.*}}: memref<64xi32>, %{{.*}}: memref<64xi32>)
// CHECK-SAME: -> (i32, i32)
// CHECK: ortumcore.cx_reduce_mac
// CHECK-NOT: conjugate
// CHECK-NOT: !ortumcore.acc

// The correlation keeps its conjugate flag across the layer.
func.func @cx_correlate_to_ortumcore(
    %real: !ondsp.acc<storage = i32, frac = 30, signed, update_overflow = saturate>,
    %imag: !ondsp.acc<storage = i32, frac = 30, signed, update_overflow = saturate>,
    %lhs: memref<64xi32>, %rhs: memref<64xi32>)
    -> (!ondsp.acc<storage = i32, frac = 30, signed, update_overflow = saturate>,
        !ondsp.acc<storage = i32, frac = 30, signed, update_overflow = saturate>) {
  %re, %im = ondsp.cx_reduce_mac %real, %imag, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    conjugate
  } : (!ondsp.acc<storage = i32, frac = 30, signed, update_overflow = saturate>,
       !ondsp.acc<storage = i32, frac = 30, signed, update_overflow = saturate>,
       memref<64xi32>, memref<64xi32>)
      -> (!ondsp.acc<storage = i32, frac = 30, signed, update_overflow = saturate>,
          !ondsp.acc<storage = i32, frac = 30, signed, update_overflow = saturate>)
  return %re, %im : !ondsp.acc<storage = i32, frac = 30, signed, update_overflow = saturate>,
                    !ondsp.acc<storage = i32, frac = 30, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @cx_correlate_to_ortumcore
// CHECK: ortumcore.cx_reduce_mac {{.*}}conjugate

// The component readout is the declared narrowing on a raw i32, not the
// 40-bit accumulator's output path, so no acc_out appears.
func.func @cx_dot_q15_readout(
    %real: !ondsp.acc<storage = i32, frac = 30, signed, update_overflow = saturate>,
    %imag: !ondsp.acc<storage = i32, frac = 30, signed, update_overflow = saturate>,
    %lhs: memref<64xi32>, %rhs: memref<64xi32>) -> (i16, i16) {
  %re, %im = ondsp.cx_reduce_mac %real, %imag, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>
  } : (!ondsp.acc<storage = i32, frac = 30, signed, update_overflow = saturate>,
       !ondsp.acc<storage = i32, frac = 30, signed, update_overflow = saturate>,
       memref<64xi32>, memref<64xi32>)
      -> (!ondsp.acc<storage = i32, frac = 30, signed, update_overflow = saturate>,
          !ondsp.acc<storage = i32, frac = 30, signed, update_overflow = saturate>)
  %exportedReal = ondsp.acc_export %re {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>, overflow = #ondsp.overflow<saturate>
  } : (!ondsp.acc<storage = i32, frac = 30, signed, update_overflow = saturate>) -> i16
  %exportedImag = ondsp.acc_export %im {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>, overflow = #ondsp.overflow<saturate>
  } : (!ondsp.acc<storage = i32, frac = 30, signed, update_overflow = saturate>) -> i16
  return %exportedReal, %exportedImag : i16, i16
}

// CHECK-LABEL: func.func @cx_dot_q15_readout
// CHECK: ortumcore.cx_reduce_mac
// CHECK: arith.shrsi %{{.*}}, %{{.*}} : i32
// CHECK: arith.trunci %{{.*}} : i32 to i16
// CHECK-NOT: ortumcore.acc_out

// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s --check-prefix=ONDSP
// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --lower-ondsp-f32-reduce-to-scalar | FileCheck %s

// The reduction reaches the same carrier the packed widths use; only the
// accumulator type follows the format instead of a declared width.
func.func @cx_dot_off(%x: memref<16xf32>, %y: memref<16xf32>) -> (f32, f32) {
  %re, %im = ondrix.cx_dot %x, %y {
    numeric = #ondsp.fp<format = f32, contract = off>,
    layout = #ondsp.cx_layout<interleaved>
  } : (memref<16xf32>, memref<16xf32>) -> (f32, f32)
  return %re, %im : f32, f32
}

// ONDSP-LABEL: func.func @cx_dot_off
// ONDSP: ondsp.cx_reduce_mac
// ONDSP-SAME: layout = #ondsp.cx_layout<interleaved>
// ONDSP-SAME: numeric = #ondsp.fp<format = f32, contract = off>
// ONDSP-NOT: ondsp.acc_zero

// A bin is two elements, so the trip count is half the operand length and
// the term is the declared pairing with nothing fused under off.
// CHECK-LABEL: func.func @cx_dot_off
// CHECK: arith.divui
// CHECK: scf.for
// CHECK: %[[XR:.*]] = memref.load %arg0[%[[RE:.*]]]
// CHECK: %[[XI:.*]] = memref.load %arg0[%[[IM:.*]]]
// CHECK: %[[YR:.*]] = memref.load %arg1[%[[RE]]]
// CHECK: %[[YI:.*]] = memref.load %arg1[%[[IM]]]
// CHECK: %[[RR:.*]] = arith.mulf %[[XR]], %[[YR]]
// CHECK: %[[II:.*]] = arith.mulf %[[XI]], %[[YI]]
// CHECK: arith.subf %[[RR]], %[[II]]
// CHECK: %[[RI:.*]] = arith.mulf %[[XR]], %[[YI]]
// CHECK: %[[IR:.*]] = arith.mulf %[[XI]], %[[YR]]
// CHECK: arith.addf %[[RI]], %[[IR]]
// CHECK-NOT: math.fma

// The conjugate term flips exactly one sign per component, and under fma the
// subtracted product rides the fused update through an exact negation.
func.func @cx_dot_conjugate_fma(%x: memref<16xf32>, %y: memref<16xf32>) -> (f32, f32) {
  %re, %im = ondrix.cx_dot %x, %y {
    numeric = #ondsp.fp<format = f32, contract = fma>,
    layout = #ondsp.cx_layout<interleaved>,
    conjugate
  } : (memref<16xf32>, memref<16xf32>) -> (f32, f32)
  return %re, %im : f32, f32
}

// CHECK-LABEL: func.func @cx_dot_conjugate_fma
// CHECK: %[[XR:.*]] = memref.load %arg0[%{{.*}}]
// CHECK: %[[XI:.*]] = memref.load %arg0[%{{.*}}]
// CHECK: %[[YR:.*]] = memref.load %arg1[%{{.*}}]
// CHECK: %[[YI:.*]] = memref.load %arg1[%{{.*}}]
// CHECK: %[[SEED:.*]] = arith.mulf %[[XR]], %[[YR]]
// CHECK: math.fma %[[XI]], %[[YI]], %[[SEED]]
// CHECK: %[[ISEED:.*]] = arith.mulf %[[XI]], %[[YR]]
// CHECK: %[[NEG:.*]] = arith.negf %[[XR]]
// CHECK: math.fma %[[NEG]], %[[YI]], %[[ISEED]]

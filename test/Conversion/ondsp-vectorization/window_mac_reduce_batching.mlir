// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" --cse --canonicalize > %t.ordered.mlir
// RUN: ondrix-opt %t.ordered.mlir --vectorize-ondsp-fixed-window-mac-reduce="vector-width=8" | FileCheck %s
// RUN: not ondrix-opt %t.ordered.mlir --vectorize-ondsp-fixed-window-mac-reduce="vector-width=1" 2>&1 | FileCheck %s --check-prefix=WIDTH
// RUN: not ondrix-opt %t.ordered.mlir --vectorize-ondsp-fixed-window-mac-reduce="vector-width=4294967296" 2>&1 | FileCheck %s --check-prefix=WIDE

// Reassociating arith.addi needs no range proof; the product carrier holding
// the exact product is the whole obligation (the pass description argues it).

// WIDTH: vector-width must be greater than one
// WIDE: vector-width must not exceed 4096

// CHECK-LABEL: func.func @lms_q15
// The guarded prehistory reduction keeps the ordered schedule: its body
// carries a boundary select the matcher does not model.
// CHECK: arith.select

// Eleven taps at width eight: one batched block, three terms left ordered.
// CHECK: %[[LAST:.*]] = arith.constant 7 : index
// CHECK: %[[END:.*]] = arith.constant 8 : index
// CHECK: %[[STEP:.*]] = arith.constant 8 : index
// CHECK: scf.for %[[BLOCK:.*]] = %{{.*}} to %[[END]] step %[[STEP]] iter_args
// The two operands walk in OPPOSITE directions, which is what no reduce_mac
// route can express: the sample span loads forward and reverses onto the
// backward walk, the coefficients load forward at the block index.
// CHECK: %[[TOP:.*]] = arith.subi %{{.*}}, %[[BLOCK]] : index
// CHECK: %[[BASE:.*]] = arith.subi %[[TOP]], %[[LAST]] : index
// CHECK: %[[SPAN:.*]] = vector.load %{{.*}}[%[[BASE]]] : memref<40xi16>, vector<8xi16>
// CHECK: vector.shuffle %[[SPAN]], %[[SPAN]] [7, 6, 5, 4, 3, 2, 1, 0] : vector<8xi16>, vector<8xi16>
// CHECK: vector.load %{{.*}}[%[[BLOCK]]] : memref<11xi16>, vector<8xi16>
// i16 x i16 is exact in the narrowed carrier; the accumulate keeps its width.
// CHECK: arith.muli %{{.*}} : vector<8xi32>
// CHECK: arith.extsi %{{.*}} : vector<8xi32> to vector<8xi64>
// CHECK: arith.addi %{{.*}} : vector<8xi64>
// CHECK: vector.reduction <add>, %{{.*}} : vector<8xi64> into i64
// CHECK: scf.for %{{.*}} = %[[END]] to %{{.*}} step %{{.*}} iter_args

func.func @lms_q15(%x: tensor<40xi16>, %d: tensor<40xi16>, %w: tensor<11xi16>)
    -> (tensor<40xi16>, tensor<11xi16>) {
  %error, %adapted = ondrix.lms %x, %d, %w {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>,
    step_size = 4096 : i64
  } : (tensor<40xi16>, tensor<40xi16>, tensor<11xi16>) -> (tensor<40xi16>, tensor<11xi16>)
  return %error, %adapted : tensor<40xi16>, tensor<11xi16>
}

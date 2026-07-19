// RUN: ondrix-opt %s --lower-rank-one-memref-copy-to-scf | FileCheck %s

func.func @dynamic_strided_copy(
    %source: memref<?xi16, strided<[?], offset: ?>>,
    %target: memref<?xi16, strided<[?], offset: ?>>) {
  memref.copy %source, %target :
      memref<?xi16, strided<[?], offset: ?>> to
      memref<?xi16, strided<[?], offset: ?>>
  return
}

// CHECK-LABEL: func.func @dynamic_strided_copy
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[LENGTH:.*]] = memref.dim %arg0, %[[C0]]
// CHECK: %[[SNAPSHOT:.*]] = memref.alloc(%[[LENGTH]]) : memref<?xi16>
// CHECK: scf.for %[[INDEX:.*]] = %[[C0]] to %[[LENGTH]] step %[[C1]] {
// CHECK:   %[[VALUE:.*]] = memref.load %arg0[%[[INDEX]]]
// CHECK:   memref.store %[[VALUE]], %[[SNAPSHOT]][%[[INDEX]]]
// CHECK: }
// CHECK: scf.for %[[INDEX:.*]] = %[[C0]] to %[[LENGTH]] step %[[C1]] {
// CHECK:   %[[VALUE:.*]] = memref.load %[[SNAPSHOT]][%[[INDEX]]]
// CHECK:   memref.store %[[VALUE]], %arg1[%[[INDEX]]]
// CHECK: }
// CHECK: memref.dealloc %[[SNAPSHOT]]
// CHECK-NOT: memref.copy

func.func @rank_two_copy(%source: memref<?x?xf32>,
                         %target: memref<?x?xf32>) {
  memref.copy %source, %target : memref<?x?xf32> to memref<?x?xf32>
  return
}

// CHECK-LABEL: func.func @rank_two_copy
// CHECK: memref.copy

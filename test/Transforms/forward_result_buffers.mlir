// RUN: ondrix-opt %s --forward-ondrix-result-buffers --split-input-file | FileCheck %s

// The out-parameter replaces the local result buffer; no copy, alloc or
// dealloc survives, and the stores land in the destination in order.
// CHECK-LABEL: func.func @forwarded(
// CHECK-SAME: %[[IN:.*]]: memref<8xi32>, %[[OUT:.*]]: memref<8xi32>)
// CHECK-NOT: memref.alloc
// CHECK: %[[V:.*]] = memref.load %[[IN]][%c0]
// CHECK: memref.store %[[V]], %[[OUT]][%c0]
// CHECK-NOT: memref.copy
// CHECK-NOT: memref.dealloc
// CHECK: return
func.func @forwarded(%in: memref<8xi32>, %out: memref<8xi32>) {
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<8xi32>
  %v = memref.load %in[%c0] : memref<8xi32>
  memref.store %v, %alloc[%c0] : memref<8xi32>
  memref.copy %alloc, %out : memref<8xi32> to memref<8xi32>
  memref.dealloc %alloc : memref<8xi32>
  return
}

// -----

// A destination that is also read elsewhere is not a plain result slot: the
// copy stays.
// CHECK-LABEL: func.func @destination_read(
// CHECK: memref.alloc
// CHECK: memref.copy
func.func @destination_read(%in: memref<8xi32>, %out: memref<8xi32>) -> i32 {
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<8xi32>
  %v = memref.load %in[%c0] : memref<8xi32>
  memref.store %v, %alloc[%c0] : memref<8xi32>
  memref.copy %alloc, %out : memref<8xi32> to memref<8xi32>
  %w = memref.load %out[%c0] : memref<8xi32>
  return %w : i32
}

// -----

// A source still read after the copy keeps its own storage.
// CHECK-LABEL: func.func @source_read_after(
// CHECK: memref.alloc
// CHECK: memref.copy
func.func @source_read_after(%in: memref<8xi32>, %out: memref<8xi32>) -> i32 {
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<8xi32>
  %v = memref.load %in[%c0] : memref<8xi32>
  memref.store %v, %alloc[%c0] : memref<8xi32>
  memref.copy %alloc, %out : memref<8xi32> to memref<8xi32>
  %w = memref.load %alloc[%c0] : memref<8xi32>
  memref.dealloc %alloc : memref<8xi32>
  return %w : i32
}

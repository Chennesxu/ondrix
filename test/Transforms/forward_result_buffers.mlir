// RUN: ondrix-opt %s --forward-ondrix-result-buffers=distinct-out-params=true --split-input-file | FileCheck %s
// RUN: ondrix-opt %s --forward-ondrix-result-buffers --split-input-file | FileCheck %s --check-prefix=UNDECLARED

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
// Without the declared calling convention nothing is forwarded.
// UNDECLARED-LABEL: func.func @forwarded(
// UNDECLARED: memref.alloc
// UNDECLARED: memref.copy
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

// -----

// The forwarded destination's descriptor pointers are marked noalias once
// the descriptor is expanded; the inputs are not.
// RUN: ondrix-opt %s --forward-ondrix-result-buffers=distinct-out-params=true --split-input-file --finalize-memref-to-llvm --convert-arith-to-llvm --convert-func-to-llvm --apply-ondrix-llvm-argument-attributes --reconcile-unrealized-casts | FileCheck %s --check-prefix=LLVM
// LLVM-LABEL: llvm.func @noalias_destination(
// LLVM-SAME: %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: i64, %{{.*}}: i64, %{{.*}}: i64, %{{.*}}: !llvm.ptr {llvm.noalias}, %{{.*}}: !llvm.ptr {llvm.noalias}, %{{.*}}: i64
// LLVM-NOT: ondrix.noalias_pointer_args
func.func @noalias_destination(%in: memref<8xi32>, %out: memref<8xi32>) {
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<8xi32>
  %v = memref.load %in[%c0] : memref<8xi32>
  memref.store %v, %alloc[%c0] : memref<8xi32>
  memref.copy %alloc, %out : memref<8xi32> to memref<8xi32>
  memref.dealloc %alloc : memref<8xi32>
  return
}

// -----

// A view of the allocation could be written after the copy, so any derived
// user refuses the forwarding even when it appears before the copy.
// CHECK-LABEL: func.func @source_viewed(
// CHECK: memref.alloc
// CHECK: memref.copy
func.func @source_viewed(%in: memref<8xi32>, %out: memref<8xi32>) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : i32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<8xi32>
  %view = memref.cast %alloc : memref<8xi32> to memref<?xi32>
  %v = memref.load %in[%c0] : memref<8xi32>
  memref.store %v, %alloc[%c0] : memref<8xi32>
  memref.copy %alloc, %out : memref<8xi32> to memref<8xi32>
  memref.store %c2, %view[%c0] : memref<?xi32>
  memref.dealloc %alloc : memref<8xi32>
  return
}

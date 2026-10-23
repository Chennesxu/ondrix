// RUN: ondrix-opt %s --pass-pipeline='builtin.module(func.func(hoist-ondrix-region-buffers))' | FileCheck %s

// A temporary in an if inside a loop moves to the entry, ahead of both.
// CHECK-LABEL: func.func @branch_in_loop(
// CHECK-NEXT: %[[T:.*]] = memref.alloc() : memref<4xf32>
// CHECK: scf.for
// CHECK: scf.if
// CHECK-NOT: memref.alloc
// CHECK: memref.store %{{.*}}, %[[T]]
// CHECK: return
func.func @branch_in_loop(%out: memref<4xf32>, %n: index, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    scf.if %c {
      %t = memref.alloc() : memref<4xf32>
      %v = memref.load %out[%c0] : memref<4xf32>
      memref.store %v, %t[%c1] : memref<4xf32>
      %w = memref.load %t[%c1] : memref<4xf32>
      memref.store %w, %out[%c1] : memref<4xf32>
    }
  }
  return
}

// A buffer yielded out of its region may outlive one execution of it.
// CHECK-LABEL: func.func @yielded(
// CHECK: scf.for
// CHECK-NEXT: memref.alloc() : memref<4xf32>
// CHECK: scf.if
// CHECK-NEXT: memref.alloc() : memref<4xf32>
func.func @yielded(%init: memref<4xf32>, %n: index, %c: i1) -> memref<4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%b = %init) -> (memref<4xf32>) {
    %next = memref.alloc() : memref<4xf32>
    memref.copy %b, %next : memref<4xf32> to memref<4xf32>
    scf.yield %next : memref<4xf32>
  }
  %s = scf.if %c -> (memref<4xf32>) {
    %t = memref.alloc() : memref<4xf32>
    scf.yield %t : memref<4xf32>
  } else {
    scf.yield %r : memref<4xf32>
  }
  return %s : memref<4xf32>
}

// A view of the buffer that leaves the region counts as the buffer leaving.
// CHECK-LABEL: func.func @view_yielded(
// CHECK: scf.if
// CHECK-NEXT: memref.alloc() : memref<8xf32>
func.func @view_yielded(%other: memref<4xf32, strided<[1], offset: ?>>, %c: i1)
    -> memref<4xf32, strided<[1], offset: ?>> {
  %r = scf.if %c -> (memref<4xf32, strided<[1], offset: ?>>) {
    %t = memref.alloc() : memref<8xf32>
    %v = memref.subview %t[2] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: 2>>
    %w = memref.cast %v : memref<4xf32, strided<[1], offset: 2>> to memref<4xf32, strided<[1], offset: ?>>
    scf.yield %w : memref<4xf32, strided<[1], offset: ?>>
  } else {
    scf.yield %other : memref<4xf32, strided<[1], offset: ?>>
  }
  return %r : memref<4xf32, strided<[1], offset: ?>>
}

// A buffer freed in its region is freed once per execution, so it stays.
// CHECK-LABEL: func.func @freed(
// CHECK: scf.for
// CHECK-NEXT: memref.alloc() : memref<4xf32>
func.func @freed(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %t = memref.alloc() : memref<4xf32>
    memref.dealloc %t : memref<4xf32>
  }
  return
}

// A dynamic size is computed in the loop and stays there.
// CHECK-LABEL: func.func @dynamic(
// CHECK: scf.for %[[I:.*]] =
// CHECK-NEXT: memref.alloc(%[[I]]) : memref<?xf32>
func.func @dynamic(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %t = memref.alloc(%i) : memref<?xf32>
    memref.dealloc %t : memref<?xf32>
  }
  return
}

// A call may keep the pointer, so each execution needs its own buffer.
// CHECK-LABEL: func.func @called(
// CHECK: scf.for
// CHECK-NEXT: memref.alloc() : memref<4xi32>
func.func private @keep(memref<4xi32>)
func.func @called(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %t = memref.alloc() : memref<4xi32>
    func.call @keep(%t) : (memref<4xi32>) -> ()
  }
  return
}

// An extracted address outlives the iteration that took it.
// CHECK-LABEL: func.func @address(
// CHECK: scf.for
// CHECK-NEXT: memref.alloc() : memref<4xi32>
func.func @address(%out: memref<?xindex>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %t = memref.alloc() : memref<4xi32>
    %p = memref.extract_aligned_pointer_as_index %t : memref<4xi32> -> index
    memref.store %p, %out[%i] : memref<?xindex>
  }
  return
}

// Parallel iterations may run at once; each keeps a private buffer.
// CHECK-LABEL: func.func @parallel(
// CHECK: scf.parallel
// CHECK: scf.for
// CHECK-NEXT: memref.alloc() : memref<1xindex>
func.func @parallel(%out: memref<?xindex>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.parallel (%i) = (%c0) to (%n) step (%c1) {
    scf.for %j = %c0 to %c1 step %c1 {
      %t = memref.alloc() : memref<1xindex>
      memref.store %i, %t[%c0] : memref<1xindex>
      %v = memref.load %t[%c0] : memref<1xindex>
      memref.store %v, %out[%i] : memref<?xindex>
    }
  }
  return
}

// An async task may still read the buffer while the next iteration writes it.
// CHECK-LABEL: func.func @task(
// CHECK: scf.for
// CHECK-NEXT: memref.alloc() : memref<1xindex>
func.func @task(%out: memref<?xindex>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %t = memref.alloc() : memref<1xindex>
    memref.store %i, %t[%c0] : memref<1xindex>
    %token = async.execute {
      %v = memref.load %t[%c0] : memref<1xindex>
      memref.store %v, %out[%i] : memref<?xindex>
      async.yield
    }
  }
  return
}

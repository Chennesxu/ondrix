// RUN: ondrix-opt %s --lower-memref-copy-to-scf --expand-strided-metadata --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: FileCheck %s < %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/memref_copy_to_scf_aot.c %t.o -o %t
// RUN: %t

// CHECK-NOT: @memrefCopy
// CHECK-NOT: memcpy

func.func @copy_right(%storage: memref<?xi32>) {
  %source = memref.subview %storage[0] [4] [1] :
      memref<?xi32> to memref<4xi32, strided<[1]>>
  %target = memref.subview %storage[1] [4] [1] :
      memref<?xi32> to memref<4xi32, strided<[1], offset: 1>>
  memref.copy %source, %target :
      memref<4xi32, strided<[1]>> to
      memref<4xi32, strided<[1], offset: 1>>
  return
}

// A static rank-two copy between two buffers takes the same two loop nests
// through a stack snapshot, and no libcall.
func.func @copy_matrix(%source: memref<2x3xi32>, %target: memref<2x3xi32>) {
  memref.copy %source, %target : memref<2x3xi32> to memref<2x3xi32>
  return
}

func.func @copy_left(%storage: memref<?xi32>) {
  %source = memref.subview %storage[1] [4] [1] :
      memref<?xi32> to memref<4xi32, strided<[1], offset: 1>>
  %target = memref.subview %storage[0] [4] [1] :
      memref<?xi32> to memref<4xi32, strided<[1]>>
  memref.copy %source, %target :
      memref<4xi32, strided<[1], offset: 1>> to
      memref<4xi32, strided<[1]>>
  return
}

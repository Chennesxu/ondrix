// RUN: ondrix-opt %s --emit-ondrix-c-entry-points --split-input-file | FileCheck %s

// One length after both pointers of the declared group; each descriptor is
// the pointer twice, offset zero, the length and stride one; the record moves
// to the wrapper and leaves the descriptor wrapper's copy.
// CHECK-LABEL: llvm.func @dot(
// CHECK-NOT: ondrix.
// CHECK-LABEL: llvm.func @_mlir_ciface_dot(
// CHECK-NOT: ondrix.
// CHECK-LABEL: llvm.func @ondrix_dot(
// CHECK-SAME: %[[A:.*]]: !llvm.ptr, %[[B:.*]]: !llvm.ptr, %[[N:.*]]: i64) -> i16
// CHECK-SAME: ondrix.c_entry = {groups = array<i64: 0, 0>, names = ["lhs", "rhs"], signature = (memref<?xi16>, memref<?xi16>) -> i16}
// CHECK: %[[MAX:.*]] = llvm.mlir.constant(9223372036854775807 : i64)
// CHECK: %[[OK:.*]] = llvm.icmp "ule" %[[N]], %[[MAX]]
// CHECK: llvm.cond_br %[[OK]], ^[[CALL:.*]], ^[[REFUSE:.*]]
// CHECK: ^[[CALL]]:
// CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i64)
// CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i64)
// CHECK: %[[R:.*]] = llvm.call @dot(%[[A]], %[[A]], %[[ZERO]], %[[N]], %[[ONE]], %[[B]], %[[B]], %[[ZERO]], %[[N]], %[[ONE]])
// CHECK: llvm.return %[[R]]
// CHECK: ^[[REFUSE]]:
// CHECK: llvm.call @puts
// CHECK: llvm.call @abort
// CHECK: llvm.unreachable
llvm.func @abort()
llvm.func @puts(!llvm.ptr)
llvm.func @dot(%a0: !llvm.ptr, %a1: !llvm.ptr, %a2: i64, %a3: i64, %a4: i64,
               %b0: !llvm.ptr, %b1: !llvm.ptr, %b2: i64, %b3: i64, %b4: i64) -> i16
    attributes {llvm.emit_c_interface, ondrix.c_entry = {groups = array<i64: 0, 0>, names = ["lhs", "rhs"],
                signature = (memref<?xi16>, memref<?xi16>) -> i16}} {
  %zero = llvm.mlir.constant(0 : i16) : i16
  llvm.return %zero : i16
}
llvm.func @_mlir_ciface_dot(%a: !llvm.ptr, %b: !llvm.ptr) -> i16
    attributes {llvm.emit_c_interface, ondrix.c_entry = {groups = array<i64: 0, 0>, names = ["lhs", "rhs"],
                signature = (memref<?xi16>, memref<?xi16>) -> i16}} {
  %zero = llvm.mlir.constant(0 : i16) : i16
  llvm.return %zero : i16
}

// -----

// A static extent needs neither a length nor a range check, and the index
// width is whatever the expanded kernel carries; separate extent groups each
// take a length right after their own pointer.
// CHECK-LABEL: llvm.func @ondrix_window(
// CHECK-SAME: %[[W:.*]]: !llvm.ptr) -> i32
// CHECK-NOT: llvm.cond_br
// CHECK: %[[FIVE:.*]] = llvm.mlir.constant(5 : i32)
// CHECK: llvm.call @window(%[[W]], %[[W]], %{{.*}}, %[[FIVE]], %{{.*}})
// CHECK-LABEL: llvm.func @ondrix_filter(
// CHECK-SAME: %{{.*}}: !llvm.ptr, %{{.*}}: i32, %{{.*}}: !llvm.ptr, %{{.*}}: i32) -> i16
llvm.func @window(%a0: !llvm.ptr, %a1: !llvm.ptr, %a2: i32, %a3: i32, %a4: i32) -> i32
    attributes {ondrix.c_entry = {groups = array<i64: -1>, names = ["w"], signature = (memref<5xi32>) -> i32}} {
  %zero = llvm.mlir.constant(0 : i32) : i32
  llvm.return %zero : i32
}
llvm.func @filter(%a0: !llvm.ptr, %a1: !llvm.ptr, %a2: i32, %a3: i32, %a4: i32,
                  %b0: !llvm.ptr, %b1: !llvm.ptr, %b2: i32, %b3: i32, %b4: i32) -> i16
    attributes {ondrix.c_entry = {groups = array<i64: 0, 1>, names = ["input", "coefficients"],
                signature = (memref<?xi16>, memref<?xi16>) -> i16}} {
  %zero = llvm.mlir.constant(0 : i16) : i16
  llvm.return %zero : i16
}

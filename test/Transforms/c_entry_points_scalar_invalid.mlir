// RUN: ondrix-opt %s --emit-ondrix-c-entry-points --split-input-file --verify-diagnostics

// Scalars cannot carry dynamic extent groups.
// expected-error @+1 {{carries a malformed C entry record}}
llvm.func @scalar_group(%a: i32) -> i32 attributes {ondrix.c_entry = {groups = array<i64: 0>, names = ["a"], signature = (i32) -> i32}} {
  llvm.return %a : i32
}

// -----

// The static/scalar sentinel is exactly -1.
// expected-error @+1 {{carries a malformed C entry record}}
llvm.func @scalar_sentinel(%a: i32) -> i32 attributes {ondrix.c_entry = {groups = array<i64: -2>, names = ["a"], signature = (i32) -> i32}} {
  llvm.return %a : i32
}

// -----

// Missing scalar names must not shift the recorded buffers.
// expected-error @+1 {{carries a malformed C entry record}}
llvm.func @missing_name(%a: i32) -> i32 attributes {ondrix.c_entry = {groups = array<i64: -1>, names = [], signature = (i32) -> i32}} {
  llvm.return %a : i32
}

// -----

// A scalar must match its actual lowered type, not only its field count.
// expected-error @+1 {{recorded C entry signature does not match the expanded function}}
llvm.func @wrong_type(%a: i64) -> i32 attributes {ondrix.c_entry = {groups = array<i64: -1>, names = ["a"], signature = (i32) -> i32}} {
  %zero = llvm.mlir.constant(0 : i32) : i32
  llvm.return %zero : i32
}

// -----

// Boolean kernels use i1; only the generated external wrapper uses i8.
// expected-error @+1 {{recorded C entry signature does not match the expanded function}}
llvm.func @wrong_bool(%a: i8) -> i1 attributes {ondrix.c_entry = {groups = array<i64: -1>, names = ["a"], signature = (i1) -> i1}} {
  %zero = llvm.mlir.constant(false) : i1
  llvm.return %zero : i1
}

// -----

// Bool is a scalar ABI conversion, not a new buffer storage format.
// expected-error @+1 {{carries a malformed C entry record}}
llvm.func @bool_buffer() attributes {ondrix.c_entry = {groups = array<i64: -1>, names = ["a"], signature = (memref<4xi1>) -> ()}} {
  llvm.return
}

// -----

// A record describes every expanded field and cannot authorize varargs.
// expected-error @+1 {{recorded C entry signature does not match the expanded function}}
llvm.func @variadic(i32, ...) -> i32 attributes {ondrix.c_entry = {groups = array<i64: -1>, names = ["a"], signature = (i32) -> i32}}

// -----

// The compact extent numbering applies across interspersed scalars.
// expected-error @+1 {{carries a malformed C entry record}}
llvm.func @skipped_group() attributes {ondrix.c_entry = {groups = array<i64: -1, 1>, names = ["a", "buffer"], signature = (i32, memref<?xi32>) -> ()}} {
  llvm.return
}

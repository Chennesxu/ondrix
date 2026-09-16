// RUN: ondrix-opt %s --emit-ondrix-c-entry-points --split-input-file --verify-diagnostics

// expected-error @+1 {{recorded C entry signature does not match the expanded function}}
llvm.func @short(%a: !llvm.ptr) -> i16
    attributes {ondrix.c_entry = {groups = array<i64: 0>, names = ["a"], signature = (memref<?xi16>) -> i16}} {
  %zero = llvm.mlir.constant(0 : i16) : i16
  llvm.return %zero : i16
}

// -----

// A recorded buffer result means the kernel returns nothing.
// expected-error @+1 {{recorded C entry signature does not match the expanded function}}
llvm.func @returns(%a0: !llvm.ptr, %a1: !llvm.ptr, %a2: i64, %a3: i64, %a4: i64,
                   %o0: !llvm.ptr, %o1: !llvm.ptr, %o2: i64, %o3: i64, %o4: i64) -> i16
    attributes {ondrix.c_entry = {groups = array<i64: -1, -1>, names = ["a", "output"], signature = (memref<4xi16>) -> memref<4xi16>}} {
  %zero = llvm.mlir.constant(0 : i16) : i16
  llvm.return %zero : i16
}

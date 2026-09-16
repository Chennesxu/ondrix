// RUN: ondrix-opt %s --emit-ondrix-c-entry-points --verify-diagnostics

// expected-error @+1 {{recorded C entry signature does not match the expanded function}}
llvm.func @short(%a: !llvm.ptr) -> i16
    attributes {ondrix.c_entry = {groups = array<i64: 0>, names = ["a"], signature = (memref<?xi16>) -> i16}} {
  %zero = llvm.mlir.constant(0 : i16) : i16
  llvm.return %zero : i16
}

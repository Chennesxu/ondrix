// RUN: ondrix-opt %s --declare-ondrix-c-entry-points --verify-diagnostics

func.func private @ondrix_taken() {
  return
}
// expected-error @+1 {{C entry point 'ondrix_taken' collides with an existing symbol}}
func.func @taken(%a: memref<?xi16>) -> i16 attributes {llvm.emit_c_interface} {
  %zero = arith.constant 0 : i16
  return %zero : i16
}

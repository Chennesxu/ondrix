// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --lower-ondsp-f32-reduce-to-scalar --convert-scf-to-cf --finalize-memref-to-llvm --convert-math-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc -ffp-contract=off %S/Inputs/cx_dot_f32_aot.c %t.o -lm -o %t
// RUN: %t

// The interleaved f32 profile has no requantization anywhere, so this gate is
// bit for bit against a reference that walks the same event graph. Both
// contract modes run because the term's second product is where they differ.

func.func @cx_dot_f32_off(%lhs: memref<16xf32>, %rhs: memref<16xf32>,
                          %out: memref<2xf32>) attributes {llvm.emit_c_interface} {
  %re, %im = ondrix.cx_dot %lhs, %rhs {
    numeric = #ondsp.fp<format = f32, contract = off>,
    layout = #ondsp.cx_layout<interleaved>
  } : (memref<16xf32>, memref<16xf32>) -> (f32, f32)
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  memref.store %re, %out[%zero] : memref<2xf32>
  memref.store %im, %out[%one] : memref<2xf32>
  return
}

func.func @cx_dot_f32_fma(%lhs: memref<16xf32>, %rhs: memref<16xf32>,
                          %out: memref<2xf32>) attributes {llvm.emit_c_interface} {
  %re, %im = ondrix.cx_dot %lhs, %rhs {
    numeric = #ondsp.fp<format = f32, contract = fma>,
    layout = #ondsp.cx_layout<interleaved>
  } : (memref<16xf32>, memref<16xf32>) -> (f32, f32)
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  memref.store %re, %out[%zero] : memref<2xf32>
  memref.store %im, %out[%one] : memref<2xf32>
  return
}

func.func @cx_corr_f32_off(%lhs: memref<16xf32>, %rhs: memref<16xf32>,
                           %out: memref<2xf32>) attributes {llvm.emit_c_interface} {
  %re, %im = ondrix.cx_dot %lhs, %rhs {
    numeric = #ondsp.fp<format = f32, contract = off>,
    layout = #ondsp.cx_layout<interleaved>,
    conjugate
  } : (memref<16xf32>, memref<16xf32>) -> (f32, f32)
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  memref.store %re, %out[%zero] : memref<2xf32>
  memref.store %im, %out[%one] : memref<2xf32>
  return
}

func.func @cx_corr_f32_fma(%lhs: memref<16xf32>, %rhs: memref<16xf32>,
                           %out: memref<2xf32>) attributes {llvm.emit_c_interface} {
  %re, %im = ondrix.cx_dot %lhs, %rhs {
    numeric = #ondsp.fp<format = f32, contract = fma>,
    layout = #ondsp.cx_layout<interleaved>,
    conjugate
  } : (memref<16xf32>, memref<16xf32>) -> (f32, f32)
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  memref.store %re, %out[%zero] : memref<2xf32>
  memref.store %im, %out[%one] : memref<2xf32>
  return
}

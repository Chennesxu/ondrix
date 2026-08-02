// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-math-to-llvm --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc -ffp-contract=off %S/Inputs/fp_scalar_dot_aot.c %t.o -lm -o %t
// RUN: %t

func.func @dot_f32_off(%lhs: f32, %rhs: f32) -> f32 {
  %result = ondrix.dot %lhs, %rhs {
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (f32, f32) -> f32
  return %result : f32
}

func.func @dot_f32_fma(%lhs: f32, %rhs: f32) -> f32 {
  %result = ondrix.dot %lhs, %rhs {
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (f32, f32) -> f32
  return %result : f32
}

func.func @dot_f32_fast(%lhs: f32, %rhs: f32) -> f32 {
  %result = ondrix.dot %lhs, %rhs {
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (f32, f32) -> f32
  return %result : f32
}

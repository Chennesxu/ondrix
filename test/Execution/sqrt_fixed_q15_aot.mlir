// RUN: ondrix-opt %s --convert-ondsp-fixed-to-scalar --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/sqrt_fixed_q15_aot.c %t.o -o %t
// RUN: %t
// RUN: ondrix-opt %s --convert-ondsp-fixed-to-scalar="sqrt-estimate" --convert-math-to-llvm --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.estimate.mlir
// RUN: ondrix-translate %t.estimate.mlir --mlir-to-llvmir > %t.estimate.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.estimate.ll -o %t.estimate.o
// RUN: cc %S/Inputs/sqrt_fixed_q15_aot.c %t.estimate.o -o %t.estimate -lm
// RUN: %t.estimate

// Direct execution gate for the full i64 domain of ondsp.sqrt_fixed under
// BOTH scalar lowerings, against an independent 32-candidate-bit unsigned
// integer square root. The harness corpus is built from algebraic
// boundaries rather than sampling: exact squares and their r^2 - 1,
// r^2 + r, r^2 + r + 1, (r+1)^2 - 1 neighbors (the nearest-rounding
// decision flips between r^2 + r and r^2 + r + 1), the i16 saturation
// boundary around 32767/32768, the 2^32 estimate-ceiling boundary, the
// 2^53 exact binary64 representability boundary that the estimate
// lowering's proof deliberately does not depend on, INT64_MAX, and
// out-of-domain negative runtime values that the deterministic zero clamp
// pins to 0.

func.func @sqrt_nearest_q15(%input: i64) -> i16 {
  %root = ondsp.sqrt_fixed %input {
    rounding = #ondsp.rounding<nearest_even>
  } : (i64) -> i16
  return %root : i16
}

func.func @sqrt_floor_q15(%input: i64) -> i16 {
  %root = ondsp.sqrt_fixed %input {
    rounding = #ondsp.rounding<toward_negative>
  } : (i64) -> i16
  return %root : i16
}

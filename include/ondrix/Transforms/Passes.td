#ifndef ONDRIX_TRANSFORMS_PASSES
#define ONDRIX_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def SpecializeOndrixConstantFir
    : Pass<"specialize-ondrix-constant-fir", "::mlir::ModuleOp"> {
  let summary = "Specialize fixed-point FIR samples with immutable coefficients";
  let description = [{
    Opt-in specialization for fixed-point `ondrix.fir` operations whose
    coefficient windows come directly from constant memref globals. Zero taps
    are removed while retaining the source update order. Symmetric
    full-product taps may be paired only for wrapping
    accumulators, where reassociation is exact modulo the accumulator width.
    Symmetric pairing remains disabled for mutable, dynamic, floating-point,
    raw-high, and saturating cases. Independent zero-tap removal may still
    apply when immutable fixed-point coefficients prove that it is exact.
  }];
  let constructor = "ondrix::createSpecializeOndrixConstantFirPass()";
  let dependentDialects = [
    "ondrix::ir::OndrixDialect",
    "ondrix::ondsp::OndspDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
  ];
  let options = [
    Option<"maxTaps", "max-taps", "int64_t", /*default=*/"64",
           "Maximum constant FIR length expanded into explicit updates">,
  ];
}

#endif // ONDRIX_TRANSFORMS_PASSES

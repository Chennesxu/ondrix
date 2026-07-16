#ifndef ONDRIX_CONVERSION_PASSES
#define ONDRIX_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertOndrixToOndsp : Pass<"convert-ondrix-to-ondsp", "::mlir::ModuleOp"> {
  let summary = "Lower ondrix algorithm intent ops to ondsp semantic ops";
  let description = [{
    Lower ondrix algorithm intent ops to ondsp semantic ops.
  }];
  let constructor = "ondrix::createConvertOndrixToOndspPass()";
  let dependentDialects = [
    "ondrix::ir::OndrixDialect",
    "ondrix::ondsp::OndspDialect",
    "mlir::arith::ArithDialect",
    "mlir::math::MathDialect",
  ];
}

def ConvertOndspToOrtumCore
    : Pass<"convert-ondsp-to-ortumcore", "::mlir::ModuleOp"> {
  let summary = "Lower ondsp semantic ops to ortumcore target semantic ops";
  let description = [{
    Lower ondsp semantic ops to ortumcore target semantic ops.
  }];
  let constructor = "ondrix::createConvertOndspToOrtumCorePass()";
  let dependentDialects = [
    "ondrix::ondsp::OndspDialect",
    "ondrix::ortumcore::OrtumCoreDialect",
    "mlir::func::FuncDialect",
  ];
}

def ConvertOrtumCoreToOndspEmulation
    : Pass<"convert-ortumcore-to-ondsp-emulation", "::mlir::ModuleOp"> {
  let summary = "Expand supported OrtumCore capabilities to public Ondsp semantics";
  let description = [{
    Fully expand the public OrtumCore accumulator subset to equivalent Ondsp
    operations. The supported subset is the signed i40/frac30 saturating
    accumulator, zero initialization, and signed Q15 full-product MAC add/sub.
    Other OrtumCore operations fail closed. Run `convert-ondsp-fixed-to-scalar`
    afterwards to obtain generic executable scalar IR.
  }];
  let constructor = "ondrix::createConvertOrtumCoreToOndspEmulationPass()";
  let dependentDialects = [
    "ondrix::ondsp::OndspDialect",
    "ondrix::ortumcore::OrtumCoreDialect",
    "mlir::arith::ArithDialect",
    "mlir::cf::ControlFlowDialect",
    "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect",
  ];
}

def LowerOndspF32ReduceToScalar
    : Pass<"lower-ondsp-f32-reduce-to-scalar", "::mlir::ModuleOp"> {
  let summary = "Lower f32 ondsp reductions to generic scalar MLIR";
  let description = [{
    Partially lower rank-1 f32 memref ondsp.reduce_mac operations to generic
    scalar MLIR. Other ondsp operations and non-memref reductions remain legal;
    this is not a final scalar legalization pass.
  }];
  let constructor = "ondrix::createLowerOndspF32ReduceToScalarPass()";
  let dependentDialects = [
    "ondrix::ondsp::OndspDialect",
    "mlir::arith::ArithDialect",
    "mlir::cf::ControlFlowDialect",
    "mlir::func::FuncDialect",
    "mlir::math::MathDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
  ];
}

def ConvertOndspFixedToScalar
    : Pass<"convert-ondsp-fixed-to-scalar", "::mlir::ModuleOp"> {
  let summary = "Fully lower supported explicit fixed-point accumulator semantics";
  let description = [{
    Fully lower supported signed fixed-point accumulator lifecycles to generic
    scalar integer operations. The current domains are Q15 full product with
    i40/frac30 accumulation, Q31 full product with i64/frac62 accumulation,
    and Q31 signed raw-high product with i40/frac30 accumulation. The pass
    converts accumulator types structurally across functions and control flow
    and fails if any unsupported ondsp operation or accumulator domain remains.
    Ondrix intent operations must be lowered first. Q15 memref reductions may be chunked by
    `vectorize-ondsp-q15-memref-reduce`. Proven exact-modulo reductions may
    then be handled by `parallelize-ondsp-q15-wrap-vector-reduce`; remaining
    fixed-length vector reductions must be normalized by
    `normalize-ondsp-q15-vector-reduce` before this final conversion.
  }];
  let constructor = "ondrix::createConvertOndspFixedToScalarPass()";
  let dependentDialects = [
    "ondrix::ondsp::OndspDialect",
    "mlir::arith::ArithDialect",
    "mlir::cf::ControlFlowDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
  ];
}

def NormalizeOndspQ15VectorReduce
    : Pass<"normalize-ondsp-q15-vector-reduce", "::mlir::ModuleOp"> {
  let summary = "Normalize fixed-length Q15 vector reductions without reassociation";
  let description = [{
    Normalize supported fixed-length Q15 vector reductions to vector full-product
    arithmetic followed by explicit accumulator updates in increasing lane
    order. The pass does not form a tree reduction and therefore preserves the
    ordered reduction contract for saturating accumulators.
  }];
  let constructor = "ondrix::createNormalizeOndspQ15VectorReducePass()";
  let dependentDialects = [
    "ondrix::ondsp::OndspDialect",
    "mlir::arith::ArithDialect",
    "mlir::vector::VectorDialect",
  ];
}

def ParallelizeOndspQ15WrapVectorReduce
    : Pass<"parallelize-ondsp-q15-wrap-vector-reduce", "::mlir::ModuleOp"> {
  let summary = "Parallelize exact-modulo Q15 vector reductions";
  let description = [{
    Reassociate a fixed-length signed Q15 vector reduction only when the
    accumulator update policy is exact modulo wrap. Products are computed
    exactly in vector i32 lanes, widened to i64, horizontally added, and then
    applied as one explicit accumulator update. Saturating reductions and
    unsupported policies remain unchanged for ordered normalization.
  }];
  let constructor = "ondrix::createParallelizeOndspQ15WrapVectorReducePass()";
  let dependentDialects = [
    "ondrix::ondsp::OndspDialect",
    "mlir::arith::ArithDialect",
    "mlir::vector::VectorDialect",
  ];
}

def VectorizeOndspQ15MemRefReduce
    : Pass<"vectorize-ondsp-q15-memref-reduce", "::mlir::ModuleOp"> {
  let summary = "Vectorize unit-stride Q15 memref reductions without reassociation";
  let description = [{
    Split supported rank-1 Q15 memref reductions into fixed-width vector
    chunks and a scalar tail. Vector chunks retain an ordered ondsp.reduce_mac
    operation for subsequent normalization, and the scalar tail uses ondsp.mac.
    The transformation preserves increasing-index accumulator update order.
    Default and representable nonnegative integer LLVM memory spaces are
    supported. Out-of-range integer spaces are rejected before LLVM lowering.
    Memory spaces that require a target-specific type-attribute conversion
    remain unchanged, as do memrefs without a statically known unit minor
    stride.
  }];
  let constructor = "ondrix::createVectorizeOndspQ15MemRefReducePass()";
  let dependentDialects = [
    "ondrix::ondsp::OndspDialect",
    "mlir::arith::ArithDialect",
    "mlir::cf::ControlFlowDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::vector::VectorDialect",
  ];
  let options = [
    Option<"vectorWidth", "vector-width", "int64_t", /*default=*/"8",
           "Number of fixed Q15 lanes per vector chunk">,
  ];
}

#endif // ONDRIX_CONVERSION_PASSES

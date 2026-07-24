#ifndef ONDRIX_CONVERSION_PASSES
#define ONDRIX_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertOndrixToOndsp : Pass<"convert-ondrix-to-ondsp", "::mlir::ModuleOp"> {
  let summary = "Lower ondrix algorithm intent ops to ondsp semantic ops";
  let description = [{
    Lower ondrix algorithm intent ops to ondsp semantic ops.
    The opt-in static CFFT Vector mode groups only independent butterflies
    within one combine stage. It preserves recursive stage order and every
    product/output requantization boundary.
  }];
  let constructor = "ondrix::createConvertOndrixToOndspPass()";
  let dependentDialects = [
    "ondrix::ir::OndrixDialect",
    "ondrix::ondsp::OndspDialect",
    "mlir::arith::ArithDialect",
    "mlir::cf::ControlFlowDialect",
    "mlir::math::MathDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect",
    "mlir::vector::VectorDialect",
  ];
  let options = [
    Option<"vectorizeStaticCfft", "vectorize-static-cfft",
           "bool", /*default=*/"false",
           "Use fixed-length Vector arithmetic for independent static CFFT stage butterflies">,
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

def LowerRankOneMemRefCopyToSCF
    : Pass<"lower-rank-one-memref-copy-to-scf", "::mlir::ModuleOp"> {
  let summary = "Lower rank-one memref copies to explicit SCF loops";
  let description = [{
    Replace ranked one-dimensional memref.copy operations with two increasing-
    index scf.for loops and a temporary buffer. The first loop snapshots the
    logical source elements and the second writes the target, preserving
    overlapping-copy semantics and arbitrary source and destination layouts.
    This avoids the external memrefCopy runtime required by MLIR 17 for dynamic
    strided descriptors. Copies of other ranks remain unchanged.
  }];
  let constructor = "ondrix::createLowerRankOneMemRefCopyToSCFPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
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
    Ondrix intent operations must be lowered first. Supported fixed-point
    constant saturating reductions may first be handled by
    `vectorize-ondsp-constant-saturating-memref-reduce`; remaining memref reductions
    may be chunked by `vectorize-ondsp-fixed-memref-reduce`.
    Proven exact-modulo reductions may then be handled by
    `parallelize-ondsp-fixed-wrap-vector-reduce`; remaining
    fixed-length vector reductions must be normalized by
    `normalize-ondsp-fixed-vector-reduce` before this final conversion.

    The opt-in `specialize-canonical-twiddles` mode recognizes compiler-visible
    packed-Q15 `+1` and `-j` constants and replaces their complex products with
    exact finite-precision identities. Runtime and other twiddles retain the
    generic widened complex-product lowering.
  }];
  let constructor = "ondrix::createConvertOndspFixedToScalarPass()";
  let options = [
    Option<"specializeCanonicalTwiddles", "specialize-canonical-twiddles",
           "bool", /*default=*/"false",
           "Use exact packed-Q15 identities for supported canonical twiddles">,
  ];
  let dependentDialects = [
    "ondrix::ondsp::OndspDialect",
    "mlir::arith::ArithDialect",
    "mlir::cf::ControlFlowDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
  ];
}

def NormalizeOndspFixedVectorReduce
    : Pass<"normalize-ondsp-fixed-vector-reduce", "::mlir::ModuleOp"> {
  let summary = "Normalize supported fixed-point vector reductions";
  let description = [{
    Normalize supported fixed-length Q15 and Q31 vector reductions to exact
    vector product arithmetic followed by explicit accumulator updates in
    increasing lane order. The pass does not form a tree reduction and
    therefore preserves the ordered reduction contract for saturating
    accumulators. Q31 raw-high products retain their explicit signed high-half
    bit selection.
  }];
  let constructor = "ondrix::createNormalizeOndspFixedVectorReducePass()";
  let dependentDialects = [
    "ondrix::ondsp::OndspDialect",
    "mlir::arith::ArithDialect",
    "mlir::vector::VectorDialect",
  ];
}

def ParallelizeOndspFixedWrapVectorReduce
    : Pass<"parallelize-ondsp-fixed-wrap-vector-reduce", "::mlir::ModuleOp"> {
  let summary = "Parallelize supported exact-modulo fixed-point vector reductions";
  let description = [{
    Reassociate a supported fixed-length signed Q15 or Q31 vector reduction
    only when the accumulator update policy is exact modulo wrap. Product terms
    are represented in i64 lanes, horizontally added modulo 2^64, and then
    applied as one explicit accumulator update. This preserves the low bits of
    every currently supported accumulator domain. Saturating reductions and
    unsupported policies remain unchanged for ordered normalization.
  }];
  let constructor = "ondrix::createParallelizeOndspFixedWrapVectorReducePass()";
  let dependentDialects = [
    "ondrix::ondsp::OndspDialect",
    "mlir::arith::ArithDialect",
    "mlir::vector::VectorDialect",
  ];
}

def VectorizeOndspConstantSaturatingMemRefReduce
    : Pass<"vectorize-ondsp-constant-saturating-memref-reduce", "::mlir::ModuleOp"> {
  let summary = "Vectorize constant saturating reductions under a no-overflow proof";
  let description = [{
    Replace supported zero-seeded saturating Q15 and Q31 full-product memref
    reductions with fixed-width Vector product chunks and horizontal sums only
    when immutable coefficients, resolved through safe complete memref views,
    and complete prefix-range analysis prove that neither the source nor
    transformed schedule can overflow. Unsupported, mutable, partial-view,
    seeded, raw-high, and unproven reductions remain unchanged for the ordered
    Vector or scalar fallback.
  }];
  let constructor = "ondrix::createVectorizeOndspConstantSaturatingMemRefReducePass()";
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
           "Number of fixed-point lanes per no-overflow horizontal chunk">,
    Option<"maxElements", "max-elements", "int64_t", /*default=*/"4096",
           "Maximum immutable coefficient count analyzed">,
    Option<"proofTraceOutput", "proof-trace-output", "std::string", /*default=*/"\"\"",
           "Optional path for an experimental JSON proof audit trace">,
  ];
}

def VerifyOndspConstantReassociationProofTrace
    : Pass<"verify-ondsp-constant-reassociation-proof-trace", "::mlir::ModuleOp"> {
  let summary = "Revalidate an experimental constant-reassociation proof trace";
  let description = [{
    Read an experimental JSON audit trace emitted by constant saturating
    reduction Vectorization. For every record, locate the original ordered
    reduction by its module walk ordinal, rederive immutable coefficient facts,
    rebuild the subject-bound prefix-range plan, and compare all numeric fields,
    coefficients, and original/reassociated prefix bounds. This pass is
    read-only and does not make transformations legal.
  }];
  let constructor = "ondrix::createVerifyOndspConstantReassociationProofTracePass()";
  let options = [
    Option<"proofTraceInput", "proof-trace-input", "std::string", /*default=*/"\"\"",
           "Path to the experimental JSON proof trace">,
    Option<"maxElements", "max-elements", "int64_t", /*default=*/"4096",
           "Maximum immutable coefficient count revalidated">,
  ];
}

def VectorizeOndspFixedMemRefReduce
    : Pass<"vectorize-ondsp-fixed-memref-reduce", "::mlir::ModuleOp"> {
  let summary = "Chunk supported unit-stride fixed-point memref reductions";
  let description = [{
    Split supported rank-1 Q15 and Q31 memref reductions into fixed-width
    vector chunks and a scalar tail. Vector chunks retain an ordered
    ondsp.reduce_mac operation for subsequent normalization, and the scalar
    tail uses ondsp.mac. The transformation preserves increasing-index
    accumulator update order.
    Only the default LLVM memory space is supported because MLIR 17 Vector to
    LLVM lowering rejects nonzero address spaces. Other valid memory spaces
    remain unchanged for scalar or target-specific lowering; invalid integer
    spaces are rejected. Memrefs without a statically known unit minor stride
    also remain unchanged.
  }];
  let constructor = "ondrix::createVectorizeOndspFixedMemRefReducePass()";
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
           "Number of fixed-point lanes per vector chunk">,
  ];
}

#endif // ONDRIX_CONVERSION_PASSES

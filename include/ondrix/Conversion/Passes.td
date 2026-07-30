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

    The opt-in sliding-window-reuse mode lowers `ondrix.moving_average`
    with an incremental running sum (add the entering sample, subtract the
    leaving one) instead of recomputing each window. The rewrite is
    authorized by the contract itself: window sums are exact i64 values
    with no per-update saturation, so the reassociation is value-neutral.
    A saturating-accumulator mean contract could not legally take this
    lowering; that contrast is the point of keeping it explicit and
    opt-in.

    The opt-in fft-loops mode lowers static CFFT/RFFT/IRFFT through a
    bit-reversal permutation and stage/butterfly loops over in-memory
    twiddle tables instead of fully unrolled SSA butterflies. Every
    butterfly executes the same scalar `ondsp.cx_butterfly` with the same
    requantization attributes in an equivalent order, so results are
    bit-identical to the default recursive lowering; only the code shape
    changes (object footprint and compile time versus unrolled constant
    propagation). It is mutually exclusive with vectorize-static-cfft.
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
    Option<"slidingWindowReuse", "sliding-window-reuse",
           "bool", /*default=*/"false",
           "Reuse exact moving-average window sums incrementally across outputs">,
    Option<"fftLoops", "fft-loops",
           "bool", /*default=*/"false",
           "Lower static FFTs with bit-reversal/stage loops over in-memory twiddle tables "
           "(bit-identical to the unrolled lowering)">,
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

    The opt-in `specialize-canonical-twiddles` mode consumes a subject-bound,
    one-shot legality plan for compiler-visible packed-Q15 `+1` and `-j`
    constants and replaces their complex products with exact finite-precision
    identities. The plan rechecks the operation, twiddle SSA value, layout,
    numeric/product policy, and both scaling boundaries. Runtime and other
    twiddles retain the generic widened complex-product lowering.

    The opt-in `sqrt-estimate` mode lowers `ondsp.sqrt_fixed` through a
    floating-point square-root estimate plus branchless exact integer
    correction instead of the default unrolled bit-by-bit root. IEEE 754
    requires a correctly rounded square root, so the truncated estimate is
    within one of the exact floor; the lowering applies two downward and
    two upward correction steps (double the required margin) before the
    declared rounding and saturation, keeping the result bit-identical to
    the contract. The mode assumes an IEEE double square root exists on
    the target; targets without one keep the default integer lowering.

    A multi-lane accumulator lowers to one storage element per lane and runs
    the identical per-lane arithmetic: the same product widths, the same
    update clamp bounds, and the same export rounding, with vector compare
    and select in place of their scalar forms. Only `acc_zero`, `mac`, and
    `acc_export` accept more than one lane; `acc_import`, `acc_add_term`,
    `mac_sub`, and `reduce_mac` fail closed on them.
  }];
  let constructor = "ondrix::createConvertOndspFixedToScalarPass()";
  let options = [
    Option<"specializeCanonicalTwiddles", "specialize-canonical-twiddles",
           "bool", /*default=*/"false",
           "Use exact packed-Q15 identities for supported canonical twiddles">,
    Option<"sqrtEstimate", "sqrt-estimate", "bool", /*default=*/"false",
           "Lower sqrt_fixed via an IEEE estimate plus exact integer correction">,
  ];
  let dependentDialects = [
    "ondrix::ondsp::OndspDialect",
    "mlir::arith::ArithDialect",
    "mlir::cf::ControlFlowDialect",
    "mlir::func::FuncDialect",
    "mlir::math::MathDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::vector::VectorDialect",
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

def VectorizeOndspFixedDecimateOutputs
    : Pass<"vectorize-ondsp-fixed-decimate-outputs", "::mlir::ModuleOp"> {
  let summary = "Batch independent decimation outputs across accumulator lanes";
  let description = [{
    Rewrite the leading full blocks of a bufferized `ondrix.fir_decimate`
    output loop to compute `vector-width` outputs at a time in a
    multi-lane accumulator, and leave the remaining outputs on the
    untouched ordered loop.

    This is a vertical transformation: the vector lanes carry independent
    outputs, not partial sums of one output. Each lane visits the same taps
    in the same increasing order and applies the same declared accumulator
    update, so no reassociation occurs and the rewrite needs no range or
    overflow proof — including for saturating accumulators. That is the
    contrast with the horizontal reduction passes, whose lanes do change the
    fold order and therefore do require proofs. The decimation contract
    permits exactly this: consumers may batch products only while preserving
    the declared ordered updates.

    The pass is opt-in and never part of a default pipeline. It fires only
    for the loop shape decimation bufferization emits with a static factor of
    two, a static coefficient count, a static output length, contiguous
    rank-1 operand layouts in the default memory space, and a single-lane
    accumulator profile whose per-lane arithmetic the scalar lowering
    implements. Every other loop is left byte-identical, so an unmatched
    program keeps the ordered schedule rather than a partially batched one.

    Per tap, one full block issues a single contiguous load of
    `factor * vector-width` elements and one even-lane shuffle, which is what
    a factor of two makes exact. The block containing the final output stays
    on the ordered loop so that contiguous load never reads past the input.

    Aliasing is a real precondition, not an aside. The rewrite defers a
    block's stores past all of that block's tap loads, so an output that
    shares storage with the input or the coefficients would be read at a
    different time than the ordered schedule reads it, and the two schedules
    would compute different values. The pass refuses every sharing it can
    decide: it resolves each operand through casts, subviews, and the other
    view-like operations to the storage it views, and declines whenever two
    of them reach the same base or read the same global.

    What that cannot decide is distinct block arguments that alias at run
    time. This shape reaches the pass from `one-shot-bufferize` with
    `bufferize-function-boundaries`, whose function-boundary ABI passes the
    input, the coefficients, and the destination-style output as separate
    memref arguments and does not promise they are disjoint. Callers of such
    a function own that disjointness, exactly as they already own it for the
    ordered lowering's own destination-style aliasing rules; this pass adds
    the further requirement that the output be disjoint from the input, which
    the ordered schedule tolerates and the batched one does not.
  }];
  let constructor = "ondrix::createVectorizeOndspFixedDecimateOutputsPass()";
  let dependentDialects = [
    "ondrix::ondsp::OndspDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::vector::VectorDialect",
  ];
  let options = [
    Option<"vectorWidth", "vector-width", "int64_t", /*default=*/"8",
           "Number of independent outputs batched per accumulator">,
  ];
}

#endif // ONDRIX_CONVERSION_PASSES

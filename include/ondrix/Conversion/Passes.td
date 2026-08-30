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

    A fixed DF-II SOS cascade emits its sections straight-line inside the
    sample loop, instead of as an inner `scf.for`, when the coefficient,
    scale, and state operands all declare the same static section count of
    at most eight. Every section keeps its declared arithmetic in the
    declared order — same macs, same accumulator, same export boundaries,
    same state write order, sections still increasing — so the straight-line
    form is the identity on the observable sequence and carries no numeric
    obligation. The motive is that constant section indices let a backend
    disambiguate the per-section state slots that the computed section
    stride hid, and keep the coefficient addressing off the index update.
    Eight is a measured bound rather than a shape limit: the straight-line
    form still pays at twelve sections and is already slower at sixteen,
    where the instruction count keeps falling but the body stops fitting
    the fetch path. A dynamic count, a count past the bound, or operands
    whose static counts disagree keep the loop and its runtime shape
    assert.
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
           "Ablation interface: force fixed-length Vector arithmetic for independent static CFFT stage butterflies (the canonical path never sets it)">,
    Option<"preserveBufferizableReductions", "preserve-bufferizable-reductions",
           "bool", /*default=*/"false",
           "Keep the algorithm operations whose reductions bufferize directly "
           "(FIR filter/decimate, conv1d, matmul, rms, dct) in contract form, "
           "so the post-bufferization schedule stage sees their reduce_mac "
           "loops instead of pre-lowered scalar bodies">,
    Option<"slidingWindowReuse", "sliding-window-reuse",
           "bool", /*default=*/"false",
           "Ablation interface: reuse exact moving-average window sums incrementally "
           "across outputs; the canonical path recomputes each window, and a "
           "floating-point average declines the reuse because the incremental form "
           "changes its event graph">,
    Option<"fftLoops", "fft-loops",
           "bool", /*default=*/"false",
           "Ablation interface: force the bit-reversal/stage-loop FFT lowering over in-memory twiddle tables, bit-identical to the unrolled default the canonical path uses">,
  ];
}

def ConvertOndspToOrtumCore
    : Pass<"convert-ondsp-to-ortumcore", "::mlir::ModuleOp"> {
  let summary = "Lower ondsp semantic ops to ortumcore target semantic ops";
  let description = [{
    Lower ondsp semantic ops to ortumcore target semantic ops.

    Accumulator export selects the target readout capability only for the
    policies the readout provably realizes, with a saturating signed i32 or
    i16 destination. `toward_negative` maps to one readout at fractional
    shifts in [0, 15]. The readout itself shifts arithmetically and saturates
    at the 32-bit storage; an i16 destination appends an explicit clamp,
    which is exact because clamping to the wider range first cannot change
    the result of the narrower clamp (for any x, both compose to the i16
    clamp of the shifted value), and for shifts of at least 8 a 40-bit
    accumulator cannot reach the 32-bit bound at all.

    `nearest_ties_positive` at shift s is realized as the floor readout at
    s-1 followed by one increment and halve, by the nested-floor identity
    floor((acc + 2^(s-1)) / 2^s) == floor((floor(acc / 2^(s-1)) + 1) / 2).
    The identity requires the s-1 readout to be exact, so the policy is
    admitted only where an accumulator of width W cannot clip the i32
    readout: s - 1 >= W - 32, i.e. s >= 9 at W = 40 (s = 0 is admitted as
    rounding-vacuous). This deliberately does NOT add half into the
    accumulator: a mac-borne 2^(s-1) saturates at the 40-bit rail, so an
    accumulator within 2^(s-1) of the positive rail would export one below
    the true value, whereas the post-readout increment sees the saturated
    accumulator exactly as the ondsp reference does. The increment and
    halve run at i64 because the readout can be exactly INT32_MAX (s = 9,
    accumulator at the rail), where an i32 increment would wrap.

    Shifts past the capability's 15 compose, up to 15 + 31: the shift-15
    floor readout is always exact from an i40 (|acc >> 15| < 2^24, no
    clip), and floor nests exactly over a base tail shift, so
    `toward_negative` appends one arithmetic tail shift; for
    `nearest_ties_positive` the half-add commutes over the shift-15 floor
    (2^15 divides 2^(s-1) whenever s > 15) and lands as one base add on
    the 25-bit readout before the same tail. Every other export policy
    fails closed.
  }];
  let constructor = "ondrix::createConvertOndspToOrtumCorePass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "ondrix::ondsp::OndspDialect",
    "ondrix::ortumcore::OrtumCoreDialect",
    "mlir::func::FuncDialect",
  ];
}

def ConvertOndspLanePairsToOrtumCore
    : Pass<"convert-ondsp-lane-pairs-to-ortumcore", "::mlir::ModuleOp"> {
  let summary = "Pair lanes = 2 ondsp accumulator webs onto the ortumcore dual-lane MAC";
  let description = [{
    Rewrite `lanes = 2` ondsp accumulator webs into pairs of ortumcore
    accumulators advanced by `ortumcore.dmac`: `acc_zero` becomes two
    `acc_init`, each broadcast-coefficient `mac` becomes one dual-lane step
    whose value halves are the extracted lanes and whose multiplier halves
    both carry the shared coefficient, and `acc_export` becomes one proven
    readout composition per lane reassembled into the vector result. The
    per-lane readout is the same composition `convert-ondsp-to-ortumcore`
    realizes for single-lane webs, and that pass carries its exactness
    argument.

    This is an opportunity pass, not a legality wall. A function is rewritten
    only when a scan proves every operation touching a qualifying pair value
    convertible: the signed i40/frac30 saturating lane domain, signed Q15
    full-product mac policies, admitted export policies, scf control flow,
    and no pair type in the function signature. Functions that fail the scan
    are left untouched so `convert-ondsp-to-ortumcore` still fails closed on
    them. The scan is load-bearing: the underlying 1:N conversion driver
    cannot convert part of a web without leaving unresolvable casts behind.
  }];
  let constructor = "ondrix::createConvertOndspLanePairsToOrtumCorePass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
    "mlir::vector::VectorDialect",
    "ondrix::ondsp::OndspDialect",
    "ondrix::ortumcore::OrtumCoreDialect",
  ];
}

def ConvertOndspCxButterflyToOrtumCore
    : Pass<"convert-ondsp-cx-butterfly-to-ortumcore", "::mlir::ModuleOp"> {
  let summary = "Select packed-target butterflies onto the ortumcore complex pair";
  let description = [{
    Rewrite each `ondsp.cx_butterfly` whose declared policies fall inside the
    packed target inventory into `ortumcore.cx_mul_conj` plus a plain
    `ortumcore.cx_bfly`: scalar i32 packed-Q15 layout, full product, product
    scale of exactly shift 15 and output scale of shift 0 or 1, both
    boundaries rounding `toward_negative` or `nearest_ties_positive` with
    wrap or saturating narrowing. The default `nearest_even` profile is not a
    target capability and is deliberately not converted.

    The target multiplies by the conjugated twiddle, so the rewrite requires
    the twiddle to be an `arith.constant` and materializes its exact
    conjugate as a new constant; a twiddle whose imaginary component is
    -32768 has no representable exact conjugate and the operation stays
    untouched. This is an opportunity pass, not a legality wall: every
    unconverted butterfly remains a generic ondsp operation with unchanged
    semantics.

    The packed-Q31 raw-high profile is selected by DECOMPOSITION rather than
    matching: the current OrtumCore surface has no direct packed-Q31
    capability, so the conversion composes the equivalent equation from the
    existing scalar capabilities. The container halves come apart with plain
    arithmetic; each cross-term pair becomes one accumulator web of
    `ortumcore.q31_mac_add`/`q31_mac_sub` steps with a shift-0
    `ortumcore.acc_out` readout; the product scale's left shift becomes a
    saturating doubling of that readout; and each stage leg becomes one
    `ortumcore.sat_shift_add`/`sat_shift_sub` carrying the declared output
    shift. Two facts make the web exact: every raw-high term is bounded by
    2^30, so a term pair stays far inside the i40 accumulator and its
    saturating updates never clamp; and reading out at shift 0 before a
    saturating doubling equals doubling the exact term sum, because clamping
    before a monotone doubling that clamps again agrees with clamping after
    — a reachable corner, since |sum| can attain 2^31. The full-product Q31
    selection stays generic.

    The pass also selects every standalone `ondsp.add_shift`/`ondsp.sub_shift`
    whose scale is exactly the scaled saturating add/sub capability - i32
    operands and result, no pre-shift, post shift inside the capability
    domain, `toward_negative`, saturating to i32 - onto
    `ortumcore.sat_shift_add`/`sat_shift_sub`. That is the same attribute
    shape the OrtumCore emulation regenerates, so selection and emulation
    round-trip exactly; any other scale stays a generic ondsp operation.

    The pass also selects every `ondsp.bitrev_add` index walk onto the
    32-bit `ortumcore.bitrev_add` progression. The composition is exact for
    any verified width: shifting both operands' low `width` bits to the top
    of an i32 makes the 32-bit reversed-carry add compute the width-bit one,
    and the modular carry out of the window lands below the extracted bits.
  }];
  let constructor = "ondrix::createConvertOndspCxButterflyToOrtumCorePass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "ondrix::ondsp::OndspDialect",
    "ondrix::ortumcore::OrtumCoreDialect",
  ];
}

def ConvertOrtumCoreToOndspEmulation
    : Pass<"convert-ortumcore-to-ondsp-emulation", "::mlir::ModuleOp"> {
  let summary = "Expand supported OrtumCore capabilities to public Ondsp semantics";
  let description = [{
    Fully expand the public OrtumCore accumulator subset to equivalent Ondsp
    operations. The supported subset is the signed i40/frac30 saturating
    accumulator, zero initialization, signed Q15 full-product MAC add/sub,
    the dual-lane step (two independent mac_add updates, expanded to
    exactly two ondsp.mac operations), and the shifted saturating readout
    (one acc_export at the i32 storage with toward_negative rounding).
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
  let summary = "Lower f32 ondsp reductions, batching declared-off products";
  let description = [{
    Partially lower rank-1 f32 memref ondsp.reduce_mac operations. Other ondsp
    operations and non-memref reductions remain legal; this is not a final
    scalar legalization pass. With `vector-width` greater than one the emitted
    body is not purely scalar - it carries one vector load and multiply per
    block - so this pass currently makes a target-width schedule choice as well
    as a lowering. That choice belongs in the schedule stage beside the other
    batchings, and moves there when the matmul and DCT producers need to share
    the mechanism.

    `vector-width` batches the PRODUCTS of a `contract = off` reduction a block
    at a time while every accumulator fold stays scalar and in index order. No
    permission is spent and no bit changes. Each lane of the vector multiply
    performs the same single round-to-nearest-even f32 multiply the scalar
    multiply of that index would perform, the products of distinct indices do
    not depend on one another, and no floating-point exception state is
    observable here, so the block is a different topological schedule of the
    event graph the contract already declares, not a different graph. A `fma`
    or `fast` contract has no separable product - it declared the multiply and
    the add to be one event - so it keeps the ordered scalar chain. The
    strictest contract is therefore the one that admits this batching, which
    is the opposite of the usual direction.
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
    "mlir::vector::VectorDialect",
  ];
  let options = [
    Option<"vectorWidth", "vector-width", "int64_t", /*default=*/"1",
           "Products per block for a declared-off reduction; 1 keeps the scalar form">,
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
           "Ablation interface: use exact packed-Q15 identities for supported canonical twiddles (the canonical path keeps the generic product)">,
    Option<"sqrtEstimate", "sqrt-estimate", "bool", /*default=*/"false",
           "Ablation interface: lower sqrt_fixed via an IEEE estimate plus exact integer correction (the canonical path uses the bit-by-bit root)">,
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

    The same proof discharges the accumulator's update guard. Authorization
    requires every prefix of the emitted schedule to fit the accumulator
    storage, so the declared saturating update is unreachable on that schedule
    and the certified loops carry the wrapping accumulator of the same
    storage, frac and signedness instead. The declared contract is unchanged:
    the reduction that was refused a proof keeps its saturating updates, and
    an accumulator whose value reaches a consumer other than `acc_export`
    keeps them as well, because only the final value is certified. Set
    `discharge-update-guard=false` to keep the guard and measure what
    spending the proof is worth.

    A certified chunk may span several machine vectors. `chunk-multiple`
    names the widest one to attempt; the pass tries the ladder downwards and
    takes the first width the analysis authorizes, so a reduction too short
    for the wide chunk falls back to a narrower certified one instead of
    losing vectorization. The obligation is not weakened by widening: every
    prefix bound is rederived for the width actually chosen, and the trace
    records that width per subject, which the replay pass reselects rather
    than trusts. What widening buys is schedule shape — one horizontal
    collapse per chunk instead of per machine vector, so the loop-carried
    recurrence holds proportionally fewer of them.
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
    Option<"chunkMultiple", "chunk-multiple", "int64_t", /*default=*/"1",
           "Widest certified chunk, in machine vectors; the ladder falls back per reduction">,
    Option<"dischargeUpdateGuard", "discharge-update-guard", "bool", /*default=*/"true",
           "Run the certified schedule on a wrapping accumulator the proof leaves unreachable">,
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
    A chunk may span several machine vectors: `chunk-multiple` names the
    widest one and the width steps down to what the reduction's static extent
    admits, because a chunk wider than the extent leaves an empty vector loop
    and moves the whole reduction into the scalar tail. Chunking is
    order-preserving at any width, so this needs no reassociation argument;
    what it changes is how often the consumer's horizontal collapse lands in
    the loop-carried recurrence. Dynamic extents keep one machine vector.
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
    Option<"chunkMultiple", "chunk-multiple", "int64_t", /*default=*/"1",
           "Widest chunk in machine vectors; steps down to the static extent">,
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

    The canonical pipeline runs this pass inside its schedule stage; the
    standalone flag remains an ablation and oracle interface. It fires only
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

    Aliasing is a separately discharged legality obligation, not an aside.
    The rewrite defers a block's stores past all of that block's tap loads,
    so an output that shares storage with the input or the coefficients would
    be read at a different time than the ordered schedule reads it, and the
    two schedules would compute different values. Nothing about the numeric
    contract covers this: it is a memory-schedule question, discharged partly
    by a static refusal here and partly by a precondition on the caller.

    The static half fails closed. Each operand is resolved through casts,
    subviews, and the other view-like operations to the storage it views, and
    the loop is refused unless BOTH resolved bases have provenance this can
    place — a function entry argument, an `alloc`, an `alloca`, or a
    `get_global`. Anything else (a memref chosen by `arith.select`, yielded
    from an `scf.if`, returned by a call, or carried as a non-entry block
    argument — a branch successor argument or a loop's iter_args, whose
    in-function predecessors can pass the same buffer twice) is refused
    outright, because such a producer may hand back storage another operand
    already names. Among placeable bases, the same value shares, two reads of
    one global share when the symbol matches, and a fresh allocation shares
    with nothing else.

    The remaining half is a PRECONDITION, not a compiler proof: distinct
    function entry arguments, and an entry argument against a global, are not
    decidable here. This shape reaches the pass from `one-shot-bufferize` with
    `bufferize-function-boundaries`, whose function-boundary ABI passes the
    input, the coefficients, and the destination-style output as separate
    memref arguments and promises nothing about their disjointness. Enabling
    this pass therefore asserts, on the caller's behalf, that the output's
    storage is disjoint from every other operand's storage for every
    argument-or-global mix the compiler could not place. The ordered schedule
    tolerates that overlap and the batched one does not, so the assertion is
    an additional obligation this pass introduces — inherited from the
    bufferization boundary, never established by it.
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

def VectorizeOndspFixedElementwiseUpdates
    : Pass<"vectorize-ondsp-fixed-elementwise-updates", "::mlir::ModuleOp"> {
  let summary = "Batch elementwise fixed-point state updates across vector lanes";
  let description = [{
    Rewrite the leading full blocks of a bufferized elementwise fixed-point
    state-update loop to update `vector-width` state elements at a time, and
    leave the remaining elements on the untouched ordered loop. The adaptive
    filter weight update is the shape this serves: at index `k`, one declared
    product of a loop-invariant step with a sample, one declared rounding
    boundary, and one declared saturating add into `state[k]`.

    This is a vertical transformation: each lane carries its own independent
    state element and applies the identical declared update — the same product,
    the same rounding boundary, the same saturating add — so no reassociation
    occurs and the rewrite needs no range or overflow proof, saturating states
    included. That is the same mechanism class as
    `vectorize-ondsp-fixed-decimate-outputs`, and the same contrast with the
    horizontal reduction passes, whose lanes do change the fold order. State to
    state there is no dependence at all: lane `k` reads and writes only
    `state[k]`, so the block's lanes are the ordered updates themselves.

    The sample walk runs BACKWARD: index `k` reads `sample[base - k]`, so one
    block of `vector-width` indices covers a contiguous span read in decreasing
    order. The block issues one contiguous load of that span and one lane
    reversal, which places on lane `k` exactly the element the ordered schedule
    gave index `k`. The span is the set the ordered block already read, so
    nothing widens and no new element is touched.

    The canonical pipeline runs this pass inside its schedule stage; the
    standalone flag remains an ablation and oracle interface. It fires only for
    a loop whose body is exactly that shape, with a static trip count, a
    loop-invariant walk base and step, contiguous statically shaped rank-1
    operand layouts in the default memory space, and no other operation in the
    body — a guarded body, a dynamic extent, a non-unit stride, a nonzero
    memory space, or one extra operation all fail closed. Every other loop is
    left byte-identical, so an unmatched program keeps the ordered schedule
    rather than a partially batched one.

    Aliasing is a separately discharged legality obligation, not an aside. The
    rewrite widens a block's loads and defers that block's stores past them, so
    an ordered store into `state[k]` that a later index's sample load would
    have observed must not exist. The state buffer is therefore required to be
    provably distinct storage from the sample buffer, by the same analysis,
    refusal set, and residual precondition the decimate batching states: the
    static half fails closed on every base it cannot place — an opaque producer
    can return storage another operand already names — and the remaining half,
    two distinct function entry arguments or an entry argument against a
    global, is a PRECONDITION this pass asserts on the caller's behalf,
    inherited from the `bufferize-function-boundaries` ABI and never
    established by it.
  }];
  let constructor = "ondrix::createVectorizeOndspFixedElementwiseUpdatesPass()";
  let dependentDialects = [
    "ondrix::ondsp::OndspDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::vector::VectorDialect",
  ];
  let options = [
    Option<"vectorWidth", "vector-width", "int64_t", /*default=*/"8",
           "Number of independent state elements updated per vector">,
  ];
}

def PairOndspFixedReductionOutputs
    : Pass<"pair-ondsp-fixed-reduction-outputs", "::mlir::ModuleOp"> {
  let summary = "Pair independent fixed reduction outputs into dual-lane "
                "accumulator webs";
  let description = [{
    Rewrites PAIRS of independent single-lane ordered fixed reductions that
    share one operand stream into one `lanes = 2` accumulator web with one
    output per lane.

    This is a vertical transformation. Each lane performs the identical
    declared update sequence on its own accumulator, in the same increasing
    index order the ordered reduction declared, so no reassociation occurs and
    the rewrite needs no range or overflow proof — including for saturating
    accumulators. That is the same mechanism class as vertical decimate output
    batching, and the same contrast with the horizontal reduction passes,
    whose lanes do change the fold order and therefore do require proofs.

    One lane per OUTPUT is the only exact pairing available. Splitting one
    output's sum across the two lanes and re-adding the halves would recombine
    through the export boundary's 32-bit saturating readout, which the declared
    contract does not admit, so that shape is never produced.

    The shared stream becomes the multi-lane mac's scalar coefficient and the
    two differing streams become the two vector lanes. When the shared stream
    was the reduction's lhs this swaps the operand roles, which is exact
    because the full Q15 product is commutative.

    Two matchers reach the paired form. The straight-line matcher pairs
    consecutive webs of one block whose differing streams are private constant
    globals, and interleaves those two tables into a single `2K`-element pair
    global. The column-loop matcher recognizes a transposed pack buffer feeding
    a loop over the packed rows, adds a pack nest that writes adjacent output
    columns interleaved, and rewrites the column loop over output pairs.

    Both interleaved buffers carry `alignment = 4`, which is what lets the
    backend merge each lane pair's two 16-bit loads into one aligned 32-bit
    load. The column-loop matcher materializes its buffer as a small aligned
    stack allocation and refuses shapes past the stack budget, because an
    unaligned or heap-resident pair buffer forfeits the merged lane load the
    pairing exists for.

    Aliasing is a separately discharged legality obligation. The rewrite defers
    the first web's store past the second web's loads, so it is refused unless
    that store's storage is provably distinct from every source the paired web
    loads: the shared stream and the differing stream of the second web. The
    static rule is the shared `mayShareStorage` placement, which fails closed on
    any base it cannot place.

    Downstream consumers keep their own legality. This pass makes a schedule
    choice and proves no target admissibility: a paired web no consumer
    implements is still a well-formed accumulator web and stays on the generic
    path. Everything the matchers cannot prove is left byte-identical — an
    unpaired leftover web, a mismatched export policy, a pack buffer with a
    user outside the recognized set, an oversize pair buffer, and an odd output
    count all keep the single-lane schedule.

    The pass is opt-in: it belongs to no registered pipeline and fires only
    under its own flag.
  }];
  let constructor = "ondrix::createPairOndspFixedReductionOutputsPass()";
  let dependentDialects = [
    "ondrix::ondsp::OndspDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::vector::VectorDialect",
  ];
}

def VectorizeOndspFpFastMemRefReduce
    : Pass<"vectorize-ondsp-fp-fast-memref-reduce", "mlir::ModuleOp"> {
  let summary = "Vectorize f32 reductions whose call sites declared the fast "
                "relaxation";
  let description = [{
    Splits a rank-1 f32 `ondsp.reduce_mac` under `contract = fast` into
    W-lane partial sums, an explicit cross-lane halving tree, and a
    scalar tail. This is the transform that consumes the R permission the
    fast contract declares: rebuilding the reduction's additive tree across
    lanes is exactly the reordering a call site authorizes by writing
    `contract = fast`, and exactly the value change the off and fma contracts
    forbid, so those two are refused here. The order-preserving batching
    pass runs first and takes the fast sites that have an output axis;
    this pass consumes the residual reductions.

    Term conservation is what makes the produced graph derivable rather than
    merely permitted. The source tree has `N + 1` leaves: the seed, which is
    the operation's initial value, and one term per index. The lane
    accumulators are seeded with the first block's W products rather than a
    synthesized `+0.0`, which is no leaf of the declared tree and is
    observable at a signed zero. The remaining blocks cover
    `[W, floor(N/W)*W)` and the scalar tail covers `[floor(N/W)*W, N)`,
    disjointly and completely, so
    every source product appears exactly once with its original operands. The
    leaves are permuted as well as regrouped — lane `i` pairs term `i` with
    term `i + W` — which is why the declared permission is a tree rebuild over
    the same indexed terms and not reparenthesization alone, a distinction a
    cancelling pair makes observable. The initial value is the cross-lane
    fold's accumulator, so it enters exactly once and nothing synthesizes a
    start value, and it stays the first leaf rather than moving among the
    terms. Below one full block there are no lanes to fill and the ordered
    schedule is kept; that branch also covers the empty reduction. The
    cross-lane fold is an explicit halving shuffle tree ending in one
    R-recording scalar fold onto the initial value. A `vector.reduction`
    would delegate the tree shape to the backend: flagless it lowers to the
    ordered `llvm.vector.reduce.fadd`, and flagging it would hand the
    backend the permission the record says the compiler spent — the emitted
    object must stay flag-free with the schedule embodying the choice. The
    rewrite reorders reads only and produces one scalar, so unlike the
    batching passes it carries no aliasing obligation.

    `interleave = C` carries C independent partial-sum vectors through the
    block loop so the loop-carried dependence is one add per C blocks
    instead of per block — the fused chain's latency is otherwise the whole
    schedule's critical path. Term conservation is unchanged in kind: each
    chain is seeded with its own real block of W products, the group loop
    covers whole groups of C blocks, the statically counted leftover blocks
    fold one per chain, and the chains merge pairwise into interior nodes of
    the same rebuilt tree before the single R-recording fold. Nothing
    synthesizes an identity and the initial still enters exactly once, at
    the fold. Chain count needs the compile-time block count, so a dynamic
    extent keeps the single chain; a static extent uses
    `min(C, blocks)`.

    `vector-width = 1` selects the same rebuild with scalar chains: the
    interleave is then the classic multi-accumulator schedule, which splits
    the loop-carried dependence and pays on a core with no usable lanes.
    Lane count and chain count are separate resources, so the ILP half of
    this rewrite is not gated on the SIMD half. A width-one rewrite requires
    a static extent and at least two effective chains, because a single
    scalar chain would spend the permission to reproduce the ordered
    schedule; the scalar chains cover every element statically, so no tail
    loop is emitted.

    Folding a term by a separate product and addition or by one fused event
    is a target question and not a legality one — both lie inside the
    declared set — so it is read from the declared `supports-vector-fma`
    capability and never from the host: an unflagged vector `llvm.fma` on a
    target without the instruction becomes one libm call per lane. The
    default is the separate product. Either way the emission carries no
    permission of its own.

    A relaxed result is never bit-pinned. Its evidence is term conservation
    checked on an integer sub-domain where every derivable rebuild is
    exact, mechanism mutations that drop, duplicate, or misindex a term, an
    audit that the permissions in the translated LLVM IR do not exceed the
    declared set and do not leak into neighbouring exact sites, executed
    divergences from the ordered fused chain, determinism, and an error
    budget against a higher-precision reference that measures this
    implementation rather than bounding the contract. The canonical pipeline
    runs this pass inside its schedule stage; the standalone flag remains an
    ablation and oracle interface. Only the default LLVM memory space and
    statically unit-strided minor dimensions are supported; other reductions
    remain unchanged for scalar lowering.
  }];
  let constructor = "ondrix::createVectorizeOndspFpFastMemRefReducePass()";
  let dependentDialects = [
    "ondrix::ondsp::OndspDialect",
    "mlir::arith::ArithDialect",
    "mlir::cf::ControlFlowDialect",
    "mlir::math::MathDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::vector::VectorDialect",
  ];
  let options = [
    Option<"vectorWidth", "vector-width", "int64_t", /*default=*/"8",
           "Number of f32 lanes per partial-sum chain; one selects scalar "
           "chains">,
    Option<"supportsVectorFma", "supports-vector-fma", "bool", /*default=*/"false",
           "Declared target capability: the target has an f32 vector "
           "fused multiply-add. Never inferred from the host">,
    Option<"interleave", "interleave", "int64_t", /*default=*/"1",
           "Number of independent partial-sum chains carried through the "
           "block loop; above one applies only to static extents">,
  ];
}

def VectorizeOndspFpFilterOutputs
    : Pass<"vectorize-ondsp-fp-filter-outputs", "mlir::ModuleOp"> {
  let summary = "Batch independent f32 filter outputs into vector lanes under "
                "the exact contracts";
  let description = [{
    Rewrites the bufferized valid-boundary f32 filter loop so that W
    independent outputs are computed in the lanes of one vector, W outputs
    per pass over the coefficients, while every remaining output keeps the
    ordered scalar schedule.

    This is the order-preserving batching axis applied to floating point.
    The off and fma contracts pin each output's dynamic floating-point event
    graph — f32 addition is not associative, so splitting one accumulation
    chain across lanes is a value-changing rewrite no proof can rescue. What
    those contracts leave open is exactly what this pass does: no output's
    chain is split or reordered; each lane runs the declared per-output
    events verbatim (the same +0.0 initial value, the same tap order, one
    product-and-accumulate — separate under off, one fused event under fma —
    per tap), and only independent outputs are laid side by side. The
    authorization is therefore structural rather than algebraic — per-lane
    event-graph identity — and it does not depend on any arithmetic law of
    the element domain, which is what makes it the same axis the fixed-point
    decimate batching uses. It does still depend on the element and
    environment semantics: a vector lane operation must agree with the
    corresponding scalar operation, which IEEE 754 elementwise vector
    arithmetic provides.

    A fast site is admitted on the same structural ground — an
    order-preserving batch is inside every declared set, so the batching
    itself never needs R. Its taps select between the two admitted members
    by the declared `supports-vector-fma` capability, and selecting the
    fused one spends F. A fast filter fold of at least four unrolled taps
    additionally rebuilds into `interleave` per-lane chains — the +0.0 seed
    staying chain zero's first leaf, every other chain seeded by its own
    real product, merged pairwise into one R-recording top fold — under the
    contract its `ondsp.reduce_mac` still declares in place; below four
    taps every tree the round-robin can build is the declared left fold,
    so the ordered form stays. Adjacent full blocks of one matmul output
    row may group into a single pass (`column-group`, default one): the row
    splats are shared, one term's group loads are contiguous in the right
    operand, and each block keeps its own accumulator, so every lane's
    event graph is the single-block one verbatim and the grouping spends
    nothing under any contract. The group width is a measured per-target
    policy: a wide out-of-order unit pays for grouping, while an in-order
    unit whose load pipe serializes paired accesses regresses under it. One shape goes further: a matmul column loop whose
    inner axis still carries its declared fast contract (stamped by the
    bufferization as an `ondsp.numeric` attribute — a declaration, not a
    spend record, so it is an authority) may rebuild that axis into
    `interleave` independent per-lane chains, the seed staying the first
    leaf of chain zero, every other chain seeded by its own real product,
    merged pairwise into one R-recording top fold; a grouped pass divides
    the chains among its blocks so the live accumulators stay bounded. The windowed-sum loop
    carries the same stamp and rebuilds the same way, its chains seeded by
    the window's own leading values since no seed is designated, and capped
    at two chains — its terms are single adds, so halving the fold is where
    the latency payoff saturates while deeper cuts feed an in-order load
    burst (a measured policy); windows
    shorter than four terms stay ordered because every tree the round-robin
    can build there is the declared left fold itself. A batched window's
    power-of-two mean divisor is realized as its exact reciprocal multiply —
    bitwise the same mean for every f32 input, so no contract distinguishes
    the two — while any other divisor keeps its division. Absence of the stamp
    reads as exact and refuses the rebuild, so a dropped attribute costs
    only the schedule. Under `row-horizontal` a stamped fast output tile is
    rewritten the other way around: the value sequence hoists once into
    W-lane segments, each output row streams one contiguous row of an
    `[output][term]` sibling of the coefficient table (the same constant,
    transposed as pure data), the segment partials fold through an explicit
    halving tree, and R is recorded once per row on the top fold. The row
    seed stays the row's own first products. This is a measured per-target
    policy: a wide unit with the whole value sequence resident in registers
    pays for streaming the table rows, while the batched tile stays the
    narrow-unit schedule. Running this pass before the horizontal fast reduction is the
    axis-selection policy: it removed the observed regression where a fast
    site with an output axis took a worse horizontal schedule. It is a
    policy, not a guarantee — no static pass order can promise relative
    speed across targets, and the fused/separate choice remains a target
    schedule policy whose payoff is target evidence. The horizontal pass
    consumes only residual reductions with no output axis.

    The canonical pipeline runs this pass inside its schedule stage; the
    standalone flag remains an ablation and oracle interface. It fires only
    for the loop shape the valid-boundary f32 filter bufferization emits,
    with static extents, contiguous rank-1 operands in the default memory
    space, and a +0.0 reduction seed. The coefficient view may also be a
    whole-length static subview at stride +-1 of a contiguous sequence —
    the convolution coefficient order — since taps are scalar-loaded; the
    touched index range is pinned against the base extent rather than
    assumed in bounds. The windowed-sum and matmul column loops and the
    transposed-table output tile (the shape the f32 DCT bufferization emits,
    each lane seeded at its own first product) are the other three admitted
    shapes, each with its own structural matcher; batching a matmul's column
    axis additionally needs a contiguous minor dimension in the right operand
    and the result. Unlike the decimate batching there is
    no widened load: at tap t the lanes for outputs m .. m+W-1 read
    input[m+t .. m+t+W-1], exactly the union of those W ordered windows, so
    every full block of W outputs is batchable and the batched schedule
    reads no element the ordered one did not.

    Aliasing is the same separately discharged obligation as in the decimate
    batching: every shape's rewrite defers a block's W stores past all of
    that block's loads (the output tile also hoists its value reads above the
    whole batched loop), so the loop is refused unless the read and output
    storage are statically distinct (same base-storage analysis, same
    fail-closed treatment of opaque provenance and non-entry block
    arguments). The remaining half is the declared function-entry
    disjointness precondition the `.ox` ABI states.
  }];
  let constructor = "ondrix::createVectorizeOndspFpFilterOutputsPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::math::MathDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::vector::VectorDialect",
  ];
  let options = [
    Option<"vectorWidth", "vector-width", "int64_t", /*default=*/"8",
           "Number of independent outputs batched per vector">,
    Option<"supportsVectorFma", "supports-vector-fma", "bool", /*default=*/"false",
           "Declared target capability: the target has an f32 vector "
           "fused multiply-add. Decides which member a fast site's taps "
           "select. Never inferred from the host">,
    Option<"interleave", "interleave", "int64_t", /*default=*/"1",
           "Number of independent per-lane chains carried over the filter "
           "fold and the column tile's unrolled inner axis; above one "
           "applies only where the loop carries the declared fast contract">,
    Option<"columnGroup", "column-group", "int64_t", /*default=*/"1",
           "Adjacent full column blocks of one matmul output row grouped "
           "into a single pass; a measured per-target schedule policy, "
           "exact under every contract">,
    Option<"rowHorizontal", "row-horizontal", "bool", /*default=*/"false",
           "Rewrite a declared-fast output tile into per-row horizontal "
           "reductions over a transposed sibling table, spending R per row; "
           "a measured per-target schedule policy">,
  ];
}

#endif // ONDRIX_CONVERSION_PASSES

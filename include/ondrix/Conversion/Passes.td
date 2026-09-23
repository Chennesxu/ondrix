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
    twiddle tables instead of fully unrolled SSA butterflies. A packed
    profile's butterfly stays the same scalar `ondsp.cx_butterfly` with the
    same requantization attributes; the interleaved f32 profile has no
    requantization boundary and emits arith/math directly in both shapes.
    Either way every element passes the identical expression in an
    equivalent order, so results are bit-identical to the default recursive
    lowering and only the code shape changes (object footprint and compile
    time versus unrolled constant propagation). The f32 shape additionally
    peels the exact unit and quarter-turn phases into their own loops,
    because those legs carry no product event by contract. It is mutually
    exclusive with vectorize-static-cfft.

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
           "(FIR filter/decimate, conv1d, matmul, rms, dct, lms) in contract form, "
           "so the post-bufferization schedule stage sees their reduce_mac "
           "loops instead of pre-lowered scalar bodies">,
    Option<"outputBatchVectorWidth", "output-batch-vector-width", "int64_t", /*default=*/"1",
           "Declared lane count the schedule stage will batch outputs at. A "
           "binary32 DCT row loop is kept in contract form only where those "
           "lanes exist to claim it; below them the scalar tensor lowering's "
           "unrolled rows are the better body and nothing downstream wants a "
           "loop">,
    Option<"slidingWindowReuse", "sliding-window-reuse",
           "bool", /*default=*/"false",
           "Ablation interface: reuse exact moving-average window sums incrementally "
           "across outputs; the canonical path recomputes each window, and a "
           "floating-point average declines the reuse because the incremental form "
           "changes its event graph">,
    Option<"fftLoops", "fft-loops",
           "bool", /*default=*/"false",
           "Ablation interface: force the bit-reversal/stage-loop FFT lowering over in-memory twiddle tables, bit-identical to the unrolled default the canonical path uses">,
    Option<"fftLoopsVectorWidth", "fft-loops-vector-width",
           "int64_t", /*default=*/"0",
           "With fft-loops, the number of consecutive butterflies of a stage "
           "computed as one fixed-length vector once the stage's half length "
           "reaches it (a power of two; 0 keeps every stage scalar). The "
           "butterflies of a stage are independent, so the lanes change no "
           "update order">,
  ];
}

def ConvertOndspToOrtumCore
    : Pass<"convert-ondsp-to-ortumcore", "::mlir::ModuleOp"> {
  let summary = "Lower ondsp semantic ops to ortumcore target semantic ops";
  let description = [{
    Lower ondsp semantic ops to ortumcore target semantic ops.

    Two MAC families reach the target accumulator: the signed Q15 full
    product (`mac_add`/`mac_sub`) and the signed Q31 raw-high product
    (`q31_mac_add`/`q31_mac_sub`), whose term is the arithmetic high half of
    the exact 32x32 product and therefore already sits at the accumulator's
    frac30, so both share one accumulator domain and one readout path. A Q31
    reduction on this profile exports at frac30 (the readout's own position);
    the target has no left-shifting readout.

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
    halve are spelled (r >> 1) + (r & 1) on the i32 readout r, which equals
    floor((r + 1) / 2) for every r including INT32_MAX (s = 9, accumulator
    at the rail), where an i32 increment would wrap.

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

def ConvertOndspLaneSplitToOrtumCore
    : Pass<"convert-ondsp-lane-split-to-ortumcore", "::mlir::ModuleOp"> {
  let summary = "Split a certified single-lane Q15 reduction across the dual-lane MAC";
  let description = [{
    Rewrite a zero-seeded, single-lane, straight-line Q15 full-product chain
    ending in one admitted export onto two target accumulators advanced by
    `ortumcore.dmac`: lane 0 takes the even-position terms and lane 1 the odd
    ones, an odd last term pairing with an exact zero product.

    The split reassociates, so it is authorized only by a certificate: every
    term multiplies a runtime value by a constant the analysis reads (a
    literal or an element of a constant global), and the prefix-range
    interval of each lane's terms fits a signed 32-bit word. Every partial
    sum of a lane lies inside that interval, because each product interval
    contains zero, so neither lane reaches the 40-bit rail, each reads out
    exactly at shift 0, and the original chain, whose prefixes are sums of
    the two lanes' prefixes, never saturates either: both compute the exact
    sum. Where the whole sum's interval also fits a word the merge is one
    add; otherwise it stays in one word through halves: `floor((a + b) / 2)` is
    `(a >> 1) + (b >> 1) + (a & b & 1)`, exact whenever both lanes fit, and
    the export composition of `convert-ondsp-to-ortumcore` runs on it with
    that value standing in for the readout at shift one, and shifted further
    for larger readout shifts, none of which can reach the readout's 32-bit
    saturation. A readout at shift 0 is admitted only when the whole sum's
    interval also fits one word.

    A dual step reads each lane pair as one word, so the runtime terms must
    read consecutive elements of one rank-1 static stream starting at an
    even index; the element contract only promises the stream is aligned to
    its element, so the rewrite tests the pointer at run time and keeps the
    ordered chain for a misaligned stream, stating the alignment it proved
    with `memref.assume_alignment` on the split path. Anything else is left
    unchanged for `convert-ondsp-to-ortumcore`.
  }];
  let constructor = "ondrix::createConvertOndspLaneSplitToOrtumCorePass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
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

def LowerMemRefCopyToSCF
    : Pass<"lower-memref-copy-to-scf", "::mlir::ModuleOp"> {
  let summary = "Lower rank-one and statically shaped memref copies to explicit SCF loops";
  let description = [{
    Replace a rank-one or a statically shaped memref.copy with increasing-
    index scf.for loops through a temporary buffer: the first nest snapshots
    the logical source elements and the second writes the target, preserving
    overlapping-copy semantics and arbitrary source and destination layouts.
    A static shape takes its snapshot on the stack at the function entry; a
    dynamic rank-one copy allocates it. No copy the canonical pipeline emits
    then reaches a runtime: neither MLIR 17's memrefCopy for dynamic strided
    descriptors nor a `memcpy` libcall, which a bare-metal target has no
    library for. A dynamically shaped copy of higher rank stays unchanged.
  }];
  let constructor = "ondrix::createLowerMemRefCopyToSCFPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
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
    Option<"wideningMultiplyLowHalves", "widening-multiply-low-halves", "bool", /*default=*/"true",
           "Declared target capability: the lane-widening integer multiply reads its operands from the low halves of the wide lanes (x86 pmuldq), so a runtime multiplicand is widened to the carrier once before its broadcast; off, the broadcast stays at the exact product width and the lanes are sign-extended afterwards, the shape a narrow-operand widening multiply (NEON smull) selects">,
  ];
  let dependentDialects = [
    "ondrix::ondsp::OndspDialect",
    "mlir::arith::ArithDialect",
    "mlir::cf::ControlFlowDialect",
    "mlir::func::FuncDialect",
    "mlir::LLVM::LLVMDialect",
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

def ForwardOndspPackedReductionOperands
    : Pass<"forward-ondsp-packed-reduction-operands", "::mlir::ModuleOp"> {
  let summary = "Read a column reduction through the source instead of a packed copy";
  let description = [{
    The matrix-product bufferization packs the right operand into a transposed
    scratch buffer so every column reduction reads a unit-stride row, the form
    the vector legality gate admits. A reduction the schedule stage left
    ordered gains nothing from that copy and pays the pack loop, the
    allocation and its release per call (a 4x16x3 Q31 product on a 128-bit
    in-order core: 48 stores and loads for 192 multiply-adds). This pass runs
    after the schedule stage and, for every packed copy whose only readers are
    still `ondsp.reduce_mac` row views, points those reductions at a strided
    column view of the original operand and deletes the copy. The fold order
    is untouched; only the addresses the terms are read from change.
  }];
  let constructor = "ondrix::createForwardOndspPackedReductionOperandsPass()";
  let dependentDialects = [
    "ondrix::ondsp::OndspDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect"
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
    and moves the whole reduction into the scalar tail. The width also steps
    down to meet a narrower vectorized writer that updates one of the operand
    buffers inside a loop the reduction itself runs in: an adapting state read
    wider than it is written cannot stay in registers across both, and the
    repack costs more than a full chunk buys. A writer outside that loop is a
    one-time fill, which dominates the reduction rather than competing with
    it, and does not narrow the chunk. Chunking is
    order-preserving at any width, so this needs no reassociation argument;
    what it changes is how often the consumer's horizontal collapse lands in
    the loop-carried recurrence. Dynamic extents keep one machine vector.
    Only the default LLVM memory space is supported because MLIR 17 Vector to
    LLVM lowering rejects nonzero address spaces. Other valid memory spaces
    remain unchanged for scalar or target-specific lowering; invalid integer
    spaces are rejected. Memrefs without a statically known unit minor stride
    also remain unchanged.

    A reversed operand is admitted, and read rather than refused. A genuine
    convolution states its kernel reversal in the LAYOUT, because this
    operation pairs both operands in increasing index order on both sides:
    the bufferized form is a stride -1 subview, which every contiguous vector
    load has to reject. Such a chunk is instead loaded forward from the view's
    source and its lanes reversed, which delivers exactly the elements the
    ordered schedule read, in the order it read them, so the admitted set
    widens with no new numeric obligation. Only a plain reversal qualifies:
    the stride must be -1 and its origin static, both reached through the
    casts bufferization inserts, and the span must lie inside a contiguous
    source. Any other stride keeps the ordered schedule.

    A wrapping reduction of a sequence against itself with full products folds
    adjacent squares in the product width before widening them, the shape x86
    selects a packed multiply-add for; `pair-fold-squares=false` widens each
    product on its own instead, which is the shape a widening accumulate
    (NEON `smull`/`saddw`) selects, because there the even/odd fold costs two
    permutes per product vector.
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
    Option<"pairFoldSquares", "pair-fold-squares", "bool", /*default=*/"true",
           "Fold adjacent squares in the product width before widening them">,
    Option<"requantizedProducts", "requantized-products", "bool", /*default=*/"true",
           "Chunk reductions whose per-term product carries a requantization; the same "
           "128-bit measurement as the output batcher's option, applied to the route that "
           "takes the sites it declines">,
  ];
}

def VectorizeOndspFixedDecimateOutputs
    : Pass<"vectorize-ondsp-fixed-decimate-outputs", "::mlir::ModuleOp"> {
  let summary = "Batch independent outputs across accumulator lanes";
  let description = [{
    Rewrite the leading full blocks of a bufferized sliding-window output
    loop -- `ondrix.fir_filter`, `ondrix.conv1d`, or `ondrix.fir_decimate`
    at factor two -- to compute a block of outputs at a time in a
    multi-lane accumulator, and leave the remaining outputs on the
    untouched ordered loop.

    The block is the widest the output count fills on a ladder of
    `chunk-multiple` machine vectors down to one, then halves of one, the rule
    the column and constant-row batchers below share: the declared width is
    the target's register, not a per-kernel choice, and a block wider than one
    vector amortizes each tap's coefficient broadcast and window load over
    more outputs (a 32-tap convolution at eight i64 lanes: 0.82x of clang's
    -O3 at one vector, 0.96x at two, parity at four on AVX2 -- while on a
    128-bit NEON file the four-vector block's accumulators spill and the
    two-vector block is the fastest of the three). The fold per output is
    unchanged, so the block width carries no numeric obligation.

    Two matrix-product shapes take the same lanes. The column loop of the
    `ondrix.matmul` bufferization, whose right operand is read through a
    transposed packed copy, is rewritten to load contiguous row segments of
    the original right operand and broadcast the left row element per term,
    which is the same value only while the packed copy is a snapshot: the
    pack nest must precede the loop in a common block and nothing from the
    pack to the loop may write memory, else the loop stays ordered. When
    every column is batched the ordered loop and the packed copy are
    removed. A column count below `vector-width` takes every column in one
    narrower block. The unrolled outputs of a constant-table product, the
    `ondrix.dct` bufferization, are grouped by consecutive output position
    under one contract; each block reads the table columns as immediates and
    the shared input one element per term, and its export, and a following
    `round_shift`, apply elementwise. Every row must certify before a
    saturating profile drops its clamp, and the i32 term group is the
    narrowest certified across the lanes.

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
    for the loop shape the window bufferizations emit with a window stride of
    one or two, a static coefficient count, a static output length, contiguous
    rank-1 sample and output layouts in the default memory space, a
    coefficient sequence read one scalar per tap (a reversed view, the
    convolution's flipped kernel, is admitted), and a single-lane accumulator
    profile whose per-lane arithmetic the scalar lowering implements. Every
    other loop is left byte-identical, so an unmatched program keeps the
    ordered schedule rather than a partially batched one.

    Per tap, one full block issues a single contiguous load of
    `factor * vector-width` elements, plus one even-lane shuffle at factor
    two. Only blocks whose last-tap load stays inside the input are batched.

    A saturating accumulator keeps its clamp in every lane unless the
    coefficients are a constant sequence for which the prefix-range analysis
    proves that no ordered prefix reaches the accumulator rail; the lanes
    then update wrapping, which is the same function on every reachable
    value, and the export keeps the declared destination policy.

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
    Option<"chunkMultiple", "chunk-multiple", "int64_t", /*default=*/"1",
           "Widest sliding-window block in machine vectors; steps down to the output count">,
    Option<"requantizedProducts", "requantized-products", "bool", /*default=*/"true",
           "Batch sites whose per-term product carries a requantization (the Q31 matmul and "
           "DCT profiles). Off, such a site keeps its ordered schedule: on a 128-bit file the "
           "rounded i64 product lanes cost more than they save (gem5 HPI, matmul4x16x3 Q31 "
           "3136 vs 1844 ordered cycles, dct64 56940 vs 47927), while at 256 bits they win">,
    Option<"maxStraightLineCoefficients", "max-straight-line-coefficients", "int64_t",
           /*default=*/"0",
           "Coefficient immediates one constant-row block may materialize before its terms "
           "become a loop over an immutable column-major table; 0 leaves every block "
           "straight-line. A transform's rows are what reach it: the block's table is "
           "`length * vector-width` wide, so a large extent grows the object with the square "
           "of itself until this bounds it">,
  ];
}

def VectorizeOndspFixedElementwiseLoops
    : Pass<"vectorize-ondsp-fixed-elementwise-loops", "::mlir::ModuleOp"> {
  let summary = "Batch a bufferized elementwise requantization loop across vector lanes";
  let description = [{
    Rewrite the leading full blocks of a bufferized elementwise loop to
    transform `vector-width` elements at a time, and leave the remaining
    elements on the untouched ordered loop. The shape this serves is the input
    requantization the Q31 `ondrix.rms` bufferization emits ahead of its
    reduction: per index, one load, one declared `ondsp.round_shift` boundary,
    and one store to a different sequence.

    This is a vertical transformation of an elementwise map: element `i` reads
    only `in[i]` and writes only `out[i]`, so the iterations are independent
    and the lanes carry the ordered iterations themselves. No value is folded
    with another, nothing is reassociated, and the rewrite therefore needs no
    range, overflow, or prefix proof at all -- it is the same mechanism class
    as `vectorize-ondsp-fixed-elementwise-updates`, and the same contrast with
    the horizontal reduction passes, whose lanes do change the fold order.
    Each declared boundary runs in the carrier it was declared in, one lane
    per ordered element, so the batched result is the ordered result element
    by element.

    It fires only for an `scf.for` from zero to a static bound in steps of one
    with no iteration arguments, whose body is exactly one `memref.load` at
    the induction variable, a chain of single-use `ondsp.round_shift`,
    `arith.extsi`, and `arith.trunci` operations, and one `memref.store` at
    the same induction variable. The operation count of the body must equal
    the matched chain, so nothing unrecognized -- and therefore no unmodelled
    side effect or second consumer -- can be hiding in the loop. Both
    sequences must be rank-1 memrefs with a static extent, a contiguous
    layout, and a memory space the Vector to LLVM lowering accepts. Every
    other loop is left byte-identical.

    Aliasing is refused the same way the other batchers refuse it: a block's
    store is deferred past that block's loads, so the two sequences must be
    distinct storage. The static half fails closed through the shared
    provenance resolution, and the residual entry-argument precondition is the
    one stated in `vectorize-ondsp-fixed-decimate-outputs`.
  }];
  let constructor = "ondrix::createVectorizeOndspFixedElementwiseLoopsPass()";
  let dependentDialects = [
    "ondrix::ondsp::OndspDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::vector::VectorDialect",
  ];
  let options = [
    Option<"vectorWidth", "vector-width", "int64_t", /*default=*/"8",
           "Number of independent elements transformed per vector">,
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

    The sample walk runs in either direction. Where index `k` reads
    `sample[base - k]` one block of `vector-width` indices covers a contiguous
    span read in decreasing order, and the block issues one contiguous load of
    that span and one lane reversal, which places on lane `k` exactly the
    element the ordered schedule gave index `k`. Where the window reaches the
    loop as a view already positioned at its first element, index `k` reads
    `sample[k]` and the same load needs no reversal at all. Either way the span
    is the set the ordered block already read, so nothing widens and no new
    element is touched.

    The batched product may run in a carrier narrower than the declared one.
    The declared carrier is sized for the ordered chain, not for a single
    product; when both factors are sign extensions, the exact product of an
    A-bit and a B-bit signed value needs A + B bits, and `ondsp.round_shift`
    is defined on the value of its input rather than on its storage width, so
    a carrier that holds the product exactly yields the same result per lane.
    Narrowing is refused whenever a factor's width is not bounded that way,
    which keeps a step of unknown magnitude on the declared carrier.

    Holding the exact product is necessary but NOT sufficient, because the
    carrier is also what the declared boundary is lowered on. A narrower one
    can push a `round_shift` out of the subset the scalar lowering implements
    and turn a program that compiles today into a legalization failure, so
    narrowing additionally requires that the narrowed carrier still admit the
    declared scale: a shift amount below its width, and a destination no wider
    than it. `pre_shift_left` is lowered inside the input carrier and would
    count against the width budget the same way, but no route executes a
    nonzero one today, so narrowing is restricted to `pre_shift_left = 0`
    rather than resting on an argument nothing runs.

    This matters beyond code size: a Q15 update batched
    at the declared 64-bit carrier needs two machine vectors per value and no
    packed 64-bit multiply or arithmetic shift exists on the common targets,
    so the unnarrowed form is slower than the ordered loop it replaces on a
    128-bit target, while the narrowed form is faster than it on both.

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

def VectorizeOndspFixedWindowMacReduce
    : Pass<"vectorize-ondsp-fixed-window-mac-reduce", "::mlir::ModuleOp"> {
  let summary = "Fold a backward-window fixed MAC reduction across vector lanes";
  let description = [{
    Fold the leading full blocks of a convolution-shaped fixed-point MAC
    reduction across `vector-width` lanes and seed the untouched ordered
    remainder with their sum. The shape this serves is the one a genuine
    convolution lowers to and which the `reduce_mac` routes cannot express:
    term `k` reads `samples[base - k]` against `coefficients[k]`, so the two
    operands walk in OPPOSITE directions while `ondsp.reduce_mac` pairs its
    operands in increasing index order on both sides.

    Reassociating the accumulate carries NO range obligation. The declared
    accumulate is `arith.addi`, and two's-complement addition is associative
    and commutative, so every fold order of the same term multiset yields the
    same wrapped value — including the orders that wrap. That is a stronger
    authorization than the horizontal `reduce_mac` routes need, and it is why
    this pass needs no never-wraps proof and no constant-coefficient
    certificate. What it does need is that the batched product carrier holds
    the EXACT product: a carrier is accepted only when it has at least both
    factors' widths, which is what makes the narrowed carrier equal to the
    declared one on every term.

    The sample walk runs backward, so a block issues one contiguous load of
    the span it already read plus one lane reversal, which places on lane `k`
    exactly the element the ordered schedule gave term `k`. The span is the
    set the ordered block already read, so nothing widens. The rewrite reads
    both operands and writes neither, and it moves no store, so unlike the
    output and state batchings it discharges no aliasing obligation at all.

    It fires only for a loop whose body is exactly that shape, with a static
    trip count, a loop-invariant walk base, contiguous statically shaped
    rank-1 operand layouts in the default memory space, and no other operation
    in the body. Every other loop is left byte-identical.
  }];
  let constructor = "ondrix::createVectorizeOndspFixedWindowMacReducePass()";
  let dependentDialects = [
    "ondrix::ondsp::OndspDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::vector::VectorDialect",
  ];
  let options = [
    Option<"vectorWidth", "vector-width", "int64_t", /*default=*/"8",
           "Number of reduction terms folded per vector">,
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

    One lane per OUTPUT is the pairing this pass produces. Splitting one
    output's sum across the two lanes and re-adding the halves recombines
    through the export boundary's 32-bit saturating readout, and that readout
    is the identity on a half that still fits the narrower width, so a
    prefix-range certificate over the even and odd subsequences makes the
    split exact as well. It is not produced here because its cost is not this
    pass's to price: the re-add spends one widening accumulate per certified
    group, and a full-scale Q15 coefficient certifies only two products per
    group, so the split is profitable only for a coefficient sequence whose
    certified group is large.

    The shared stream becomes the multi-lane mac's scalar coefficient and the
    two differing streams become the two vector lanes. When the shared stream
    was the reduction's lhs this swaps the operand roles, which is exact
    because the full Q15 product is commutative.

    Three matchers reach the paired form. The straight-line matcher pairs
    consecutive webs of one block whose differing streams are private constant
    globals, and interleaves those two tables into a single `2K`-element pair
    global. The column-loop matcher recognizes a transposed pack buffer feeding
    a loop over the packed rows, adds a pack nest that writes adjacent output
    columns interleaved, and rewrites the column loop over output pairs. The
    window-loop matcher recognizes a loop of consecutive outputs over sliding
    windows of one stream against a loop-invariant coefficient stream — the
    valid-boundary FIR on the bufferized route — and pairs outputs `n` and
    `n + 1` with the roles turned around: the `K + 1` samples both windows
    span become the scalar coefficient, and the lanes read a pair buffer whose
    slot `2j` holds tap `j` and slot `2j + 1` tap `j - 1`, with `c[-1]` and
    `c[K]` the exact zero. A step whose product is exactly zero leaves the
    accumulator unchanged under every declared overflow mode, so each lane
    still performs its declared `K` updates in increasing tap order with one
    identity step interposed, and the pairing stays exact. The pair buffer is
    packed once before the loop, so the coefficient stream must be provably
    distinct from the destination; an odd last output stays on the original
    single-lane loop, marked `ondsp.pairing_remainder` so the unrolling passes
    keep it a loop.

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
    so the ordered form stays. Columns a matmul row's full blocks leave over,
    two or more of them, take one padded block of the next power-of-two
    width: its surplus lanes read the following row of the contiguous right
    operand for every term but the last, whose lanes are assembled, and only
    the real columns are stored, so no ordered column remains (a 4x16x3
    product runs one four-lane block per row where it ran a two-lane block
    and an ordered column). Adjacent full blocks of one matmul output
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

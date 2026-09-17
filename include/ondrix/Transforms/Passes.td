#ifndef ONDRIX_TRANSFORMS_PASSES
#define ONDRIX_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def SpecializeOndrixConstantFir
    : Pass<"specialize-ondrix-constant-fir", "::mlir::ModuleOp"> {
  let summary = "Specialize fixed-point FIR samples with immutable coefficients";
  let description = [{
    Opt-in specialization for fixed-point `ondrix.fir` operations whose
    coefficient windows resolve to complete constant memref globals through
    direct handles or proven full-range unit-stride views. Zero taps are
    removed while retaining the source update order. Symmetric
    full-product taps may be paired for wrapping accumulators, where
    reassociation is exact modulo the accumulator width. Saturating
    accumulators may be paired only when complete prefix-range analysis proves
    that neither the source nor reassociated schedule can overflow. Symmetric
    pairing remains disabled for mutable, dynamic, floating-point, and
    raw-high cases. Independent zero-tap removal may still apply when immutable
    fixed-point coefficients prove that it is exact.
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

def TileOndrixFirFilter
    : Pass<"tile-ondrix-fir-filter", "::mlir::ModuleOp"> {
  let summary = "Tile FIR filters along the output axis";
  let description = [{
    Opt-in output-axis tiling for tensor `ondrix.fir_filter` operations. The
    pass delegates loop construction and destination insertion to MLIR's SCF
    TilingInterface implementation. Valid-boundary tiles read the corresponding
    input halo. Full-boundary tiles retain the complete input and carry their
    global output origin. Every tile retains the complete coefficient tensor;
    the ordered tap reduction remains inside the tiled operation and is not
    reassociated.
  }];
  let constructor = "ondrix::createTileOndrixFirFilterPass()";
  let dependentDialects = [
    "ondrix::ir::OndrixDialect",
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::cf::ControlFlowDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect",
  ];
  let options = [
    Option<"tileSize", "tile-size", "int64_t", /*default=*/"4",
           "Number of output samples computed by each tile">,
  ];
}

def DecomposeOndrixFirStream
    : Pass<"decompose-ondrix-fir-stream", "::mlir::ModuleOp"> {
  let summary = "Decompose FIR stream chunks into tensor and full-output FIR operations";
  let description = [{
    Opt-in structural decomposition for tensor `ondrix.fir_stream`. The pass
    materializes the chronological extended sequence `concat(state, input)`,
    computes a non-empty chunk with a valid-boundary `ondrix.fir_filter`, and
    extracts the suffix that becomes the next state. Empty chunks preserve
    state without invoking a valid FIR filter. Numeric update and export
    policies are copied unchanged, so later bufferization and fixed-point
    Vector passes can reuse the existing full-output FIR implementation.
  }];
  let constructor = "ondrix::createDecomposeOndrixFirStreamPass()";
  let dependentDialects = [
    "ondrix::ir::OndrixDialect",
    "mlir::arith::ArithDialect",
    "mlir::cf::ControlFlowDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect",
  ];
}

def EvaluateOndrixFirDesign
    : Pass<"evaluate-ondrix-fir-design", "::mlir::ModuleOp"> {
  let summary = "Evaluate FIR design intent into constant Q15 coefficients";
  let description = [{
    Opt-in compile-time evaluation for `ondrix.window_hamming` and
    `ondrix.fir_design_windowed_sinc`. Each design operation is replaced by
    an `arith.constant` dense coefficient tensor produced by binary64
    evaluation of the real-valued contract equation and a single
    round-half-even signed Q1.15 quantization with declared, counted
    saturation. Evaluation fails closed with an error on the operation when
    any coefficient lies closer than `2^-20` Q15 LSB to a rounding tie, so
    an emitted table provably matches the real-valued definition under the
    declared binary64/libm evaluation error budget (documented at more than
    three orders of magnitude below the guard) and is exactly symmetric. The
    replacement constant carries an `ondrix.design_provenance` dictionary
    attribute recording the design kind, response, cutoff, and saturation
    count; downstream bufferization materializes it as a constant global
    that the existing constant-coefficient analysis and proof-gated
    transformations consume unchanged.
  }];
  let constructor = "ondrix::createEvaluateOndrixFirDesignPass()";
  let dependentDialects = [
    "ondrix::ir::OndrixDialect",
    "mlir::arith::ArithDialect",
  ];
}

def FuseOndrixElementwiseChains
    : Pass<"fuse-ondrix-elementwise-chains", "::mlir::ModuleOp"> {
  let summary = "Collapse unary elementwise chains proven equivalent by exhaustive certificate";
  let description = [{
    Opt-in rewrite of a two-operation chain of unary Q1.15 elementwise members
    (`abs`, `negate`, `offset`, `shift`) into one operation, or into the input
    when the pair is the identity. Every one of these collapses is a
    real-arithmetic identity and none of them is unconditionally a contract
    identity, so a candidate is authorized only by evaluating both programs on
    all 65536 possible i16 inputs and finding them bit-identical.

    The verdicts the certificate produces are not uniform, and that is the
    point of running it per candidate rather than per rule:

    ```text
    negate(negate(x))      identity under wrap; under saturate -32768 -> -32767
    offset(offset(x,a),b)  one offset under wrap; under saturate a=+1000 then
                           b=-1000 clamps at 32767 and loses 1000
    shift(shift(x,-a),-b)  one shift under toward_negative and toward_zero;
                           under a nearest rule the double rounding
                           diverges — with one exception the certificate
                           found and the rule-of-thumb misses: nearest_even
                           at amounts -1,-1 composes exactly (ties_positive
                           does not even there). Same double-rounding
                           structure as the round_shift cascade study
    shift(shift(x,a),b)    one shift for two left shifts under either mode
    abs(negate(x))         abs under both modes
    ```

    A binary member never enters a chain: `mult` has a 2^32 input domain and
    neither operand is known at compile time, so the enumeration that decides
    every case above cannot be run at all. That is a structural refusal rather
    than a failed certificate, and it is why the constant-operand `gain` has a
    merge pass while `mult` has none.

    An intermediate value with another use, a mismatched numeric policy, or a
    merged parameter outside the operation's declared range leaves the chain
    untouched; the safe state is the uncollapsed program. A rewritten
    operation records the pair and the certificate domain in an
    `ondrix.elementwise_fusion_provenance` attribute.
  }];
  let constructor = "ondrix::createFuseOndrixElementwiseChainsPass()";
  let dependentDialects = [
    "ondrix::ir::OndrixDialect",
  ];
}

def MergeOndrixGainCascades
    : Pass<"merge-ondrix-gain-cascades", "::mlir::ModuleOp"> {
  let summary = "Merge gain cascades proven equivalent by exhaustive certificate";
  let description = [{
    Opt-in rewrite of `ondrix.gain(ondrix.gain(x, g1), g2)` cascades into a
    single gain carrying the round-half-even quantized constant product
    `q15(g1 * g2 / 2^15)`. Merging two Q1.15 requantization boundaries into
    one is NOT a real-arithmetic identity under the contract — most
    constant pairs produce a bit-different program — so each candidate is
    authorized only by an exhaustive compile-time certificate: both
    programs are evaluated on every one of the 65536 possible i16 inputs
    and the rewrite proceeds only when they are bit-identical everywhere.
    Cascades whose certificate fails, whose intermediate value has other
    uses, or whose numeric policies differ are left untouched (the safe
    state is the unmerged program). A merged operation records the
    original constants and the certificate domain in an
    `ondrix.gain_merge_provenance` attribute.
  }];
  let constructor = "ondrix::createMergeOndrixGainCascadesPass()";
  let dependentDialects = [
    "ondrix::ir::OndrixDialect",
  ];
}

def FuseOndrixGainIntoFir
    : Pass<"fuse-ondrix-gain-into-fir", "::mlir::ModuleOp"> {
  let summary = "Fuse a constant gain into constant FIR taps under a per-tap certificate";
  let description = [{
    Opt-in rewrite of `ondrix.fir_filter(ondrix.gain(x, g), h, init)` into the
    same filter reading `x` directly with rewritten constant taps `h'`,
    deleting the gain's per-element Q1.15 requantization boundary. Scaling a
    filter's input by `g` and scaling its taps by `g` are the same operation in
    real arithmetic; under the contract they are not, because the gain rounds
    and saturates every sample before the taps ever see it.

    Each candidate is therefore authorized only by an exhaustive per-tap
    certificate. For every tap `i` and every one of the 65536 possible i16
    values `x`, the exact products entering the accumulator must be identical:

    ```text
    applyGainQ15(x, g, rule) * h[i] == x * h'[i]
    ```

    When every tap certifies, the two programs push the identical ordered
    sequence of accumulator terms for any input, so the rewrite is bit-exact
    independently of accumulator width, update overflow policy, and export
    policy — none of those can distinguish programs whose term sequences agree
    term by term. Because a product selection is a deterministic function of
    that exact product, the certificate covers `full` and `high_raw` alike.

    The certificate is evaluated under the DECLARED tie rule and the pass
    admits only the two rules `ondrix.gain` itself admits. A gain with more
    than one use, non-constant or non-i16 taps, a mismatched numeric policy,
    or a single failing tap leaves the whole filter untouched: the unfused
    program is the safe state and no diagnostic is emitted. A fused filter
    records the gain constant, the tie rule, and the certificate domain in an
    `ondrix.gain_fusion_provenance` attribute.

    The certified set is nearly empty, and that is the experimental result.
    A nonzero tap forces `applyGainQ15(x, g, rule) = x * h'[i] / h[i]` for all
    `x`, an exact linear identity that Q1.15 requantization satisfies for no
    representable nonzero gain — the unit gain `2^15` is outside `[-32768,
    32767]`. Even `g = -32768`, plain negation, fails: with `h' = -h` it agrees
    on 65535 of 65536 inputs and diverges only at `x = -32768`, where the gain
    saturates to 32767 instead of 32768. Exhaustive sweeps over the whole gain
    domain confirm the only certifying candidates are the degenerate ones,
    `g = 0` and all-zero taps, and that legality here does not depend on the
    tie rule.

    `record-refusals` is a diagnostic mode, off by default. It attaches an
    `ondrix.gain_fusion_refusal` attribute — the first failing tap and its
    first witness input with the two divergent exact products — to filters the
    certificate rejects, without rewriting them.
  }];
  let constructor = "ondrix::createFuseOndrixGainIntoFirPass()";
  let dependentDialects = [
    "ondrix::ir::OndrixDialect",
    "mlir::arith::ArithDialect",
  ];
  let options = [
    Option<"recordRefusals", "record-refusals", "bool", /*default=*/"false",
           "Attach the first failing tap and its witness input to refused filters">,
  ];
}

def ForwardOndrixInsertExtract
    : Pass<"forward-ondrix-insert-extract", "::mlir::ModuleOp"> {
  let summary = "Forward constant-index tensor reads through insert chains";
  let description = [{
    Opt-in producer-consumer fusion for staged tensor consumers. A staged
    algorithm lowering assembles an intermediate tensor with a chain of
    constant-index `tensor.insert` operations into `tensor.empty`, and the
    next stage reads it back with constant-index `tensor.extract`. This
    LLVM release has no extract-of-insert folding at all — upstream folds
    inspect only splats, from_elements, and constants — so every such
    intermediate stays fully materialized through lowering.

    This pass walks the insert chain from the extract toward the base and
    replaces the extract with the scalar of the first insert whose constant
    indices all equal the extract's. That scalar and the read are the same
    SSA value, so this is value-identity forwarding: it moves no numeric
    boundary, reorders no accumulator update, and carries no legality
    obligation beyond index equality. Once every read of the intermediate is
    forwarded, the inserts and the `tensor.empty` become dead and ordinary
    dead-code elimination removes the intermediate tensor.

    The walk refuses and leaves the extract untouched when any compared index
    is not a compile-time constant, when the ranks differ, or when it reaches
    a producer that is not a `tensor.insert` without having found a matching
    index. In particular a read that resolves to `tensor.empty` is never
    forwarded: an unmatched read of uninitialized storage is an upstream
    defect, not a value this pass may invent.
  }];
  let constructor = "ondrix::createForwardOndrixInsertExtractPass()";
  let dependentDialects = [
    "mlir::tensor::TensorDialect",
  ];
}

def ConvertOndrixStaticResultsToOutParams
    : Pass<"convert-ondrix-static-results-to-out-params", "::mlir::ModuleOp"> {
  let summary = "Pass statically shaped memref results as trailing out-parameters";
  let description = [{
    `buffer-results-to-out-params` appends every memref result of every
    function as an out-parameter and fails at a call whose result has a
    dynamic shape, since no caller could size that buffer. The destination
    the caller supplies is the calling convention DSP libraries use, so the
    canonical pipeline adopts it wherever a caller can supply the storage:
    this pass applies the upstream rewrite to the functions whose memref
    results all have static shapes, and leaves a function with a dynamically
    shaped result allocating that result itself under the descriptor-returning
    convention. The whole-buffer copy the rewrite inserts before each return
    is what `forward-ondrix-result-buffers` then removes.
  }];
  let constructor = "ondrix::createConvertOndrixStaticResultsToOutParamsPass()";
  let dependentDialects = ["mlir::memref::MemRefDialect"];
}

def DeclareOndrixArgumentsReadOnly
    : Pass<"declare-ondrix-arguments-read-only", "::mlir::ModuleOp"> {
  let summary = "Declare every tensor argument read-only for bufferization";
  let description = [{
    Marks each tensor argument of each function `bufferization.writable =
    false`. One-shot bufferization treats every function argument as
    writable, so a kernel that updates its state in place, the recursive
    filters and the adaptive ones, would write its next state into the
    caller's input buffer and then copy it to the result. A source tensor
    argument is a value and the C entry declares its pointer `const`, so the
    kernel must never write an input: with the argument declared read-only,
    bufferization copies it into a buffer of the kernel's own before the
    first write, and the result forwarding then makes that buffer the
    caller's output. Runs before `one-shot-bufferize` in the canonical
    pipeline.
  }];
  let constructor = "ondrix::createDeclareOndrixArgumentsReadOnlyPass()";
  let dependentDialects = ["mlir::bufferization::BufferizationDialect"];
}

def ForwardOndrixResultBuffers
    : Pass<"forward-ondrix-result-buffers", "::mlir::ModuleOp"> {
  let summary = "Write a function's result directly into its out-parameter";
  let description = [{
    After `buffer-results-to-out-params` a bufferized function still fills a
    local allocation and copies it into the out-parameter at the end, so a
    kernel whose caller supplies the destination pays an allocation and a
    whole-buffer copy per call. This pass replaces such an allocation by the
    out-parameter itself when the copy is the only use of that parameter, both
    buffers have the same static identity-layout type, the allocation is
    defined in the function's entry block, every use of the allocation is a
    plain load, store, copy or deallocation (no view, cast, address
    extraction or call through which it could be reached again), and nothing
    uses it after the copy other than its deallocation; the copy, the
    allocation and the deallocation are erased.

    Writing the destination before the inputs are fully read is the original
    program only if the destination overlaps no storage the function reads or
    writes by any other route. That is a calling convention, declared with
    `distinct-out-params`: every out-parameter is a caller buffer disjoint
    from every other argument and from every global the function reaches.
    Without the declaration the pass forwards nothing, and a function whose
    memory accesses do not all root at its arguments, its local allocations
    or constant globals (a call, an unknown effect, a mutable global) is left
    alone even under the declaration, since the convention cannot speak for
    storage it does not name. The same declaration authorizes recording the
    destination's pointers for `apply-ondrix-llvm-argument-attributes`.
  }];
  let options = [
    Option<"distinctOutParams", "distinct-out-params", "bool", /*default=*/"false",
           "Declares that every out-parameter is a caller buffer disjoint from "
           "every other argument and from every global the function reaches; "
           "required for any forwarding to happen">,
  ];
  let constructor = "ondrix::createForwardOndrixResultBuffersPass()";
  let dependentDialects = [
    "mlir::cf::ControlFlowDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect",
  ];
}

def ApplyOndrixLlvmArgumentAttributes
    : Pass<"apply-ondrix-llvm-argument-attributes", "::mlir::ModuleOp"> {
  let summary = "Attach recorded argument attributes after descriptor expansion";
  let description = [{
    `convert-func-to-llvm` expands a memref argument into its descriptor
    fields and drops the argument's attributes on the way, so a fact known at
    the memref level about one buffer cannot be stated to LLVM directly.
    `forward-ondrix-result-buffers` records on the function the expanded
    positions of the forwarded destination's two pointers; this pass, run
    after the conversion, marks those `llvm.func` arguments `llvm.noalias`
    and removes the record. The fact is the one the forwarding relied on: the
    destination stands for a result the function used to allocate itself,
    which aliased no other argument. Functions without the record are left
    untouched.
  }];
  let constructor = "ondrix::createApplyOndrixLlvmArgumentAttributesPass()";
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}

def DeclareOndrixCEntryPoints
    : Pass<"declare-ondrix-c-entry-points", "::mlir::ModuleOp"> {
  let summary = "Record which exported kernels receive a plain-pointer C entry point";
  let description = [{
    A kernel exported through `llvm.emit_c_interface` is reachable from C
    only through MLIR's ranked-memref descriptor convention, whose field
    layout and index width belong to no published ABI. This pass records, on
    every exported function whose signature the plain entry can carry, the
    bufferized signature that `emit-ondrix-c-entry-points` later turns into
    an `ondrix_<kernel>` wrapper. Every parameter is an identity-layout memref
    of a fixed-width integer or binary floating-point element type, either
    rank 1 or statically shaped; the results are either one scalar or
    statically shaped memrefs of the same kind, which
    `convert-ondrix-static-results-to-out-params` later passes as trailing
    out-parameters. A dynamically shaped result, a strided view or a
    dynamic extent above rank 1 gets no entry; the descriptor entry points
    are untouched either way.

    Parameter names are read from the arguments' `NameLoc` locations, which a
    frontend sets and nothing has to drop later; a result is `output`, or
    `output<i>` when there are several, kept apart from the parameter names.
    Two dynamic parameters an `ondsp.reduce_mac` pairs read equal-length
    windows by that operation's contract, so they form one extent group and
    the entry takes one length for it; the kernel's own equal-length assertion
    then holds by construction. The pass therefore runs right after
    bufferization, while the reductions are still `reduce_mac`. A kernel whose
    entry name is already a symbol of the module is an error, never a silent
    rename.
  }];
  let constructor = "ondrix::createDeclareOndrixCEntryPointsPass()";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::LLVM::LLVMDialect",
                           "mlir::memref::MemRefDialect", "ondrix::ondsp::OndspDialect"];
}

def EmitOndrixCEntryPoints
    : Pass<"emit-ondrix-c-entry-points", "::mlir::ModuleOp"> {
  let summary = "Build the plain-pointer C entry points from their recorded signatures";
  let description = [{
    After `convert-func-to-llvm` a recorded kernel is an `llvm.func` whose
    memref parameters, the recorded inputs followed by the results that
    became out-parameters, have expanded into descriptor fields (allocated
    and aligned pointers, offset, one size and one stride per dimension). For
    each record this pass adds `llvm.func @ondrix_<kernel>` taking one pointer
    per buffer and, after the last parameter of each extent group, one length
    in the index width the conversion chose; it builds the descriptors (both
    pointers the argument, offset zero, the static extents or the length as
    sizes, row-major strides) and calls the kernel, returning its scalar or
    nothing. A length whose buffer would not fit the signed range of the index
    type is refused before any memory is touched, with the message-then-abort
    convention the shape assertions already use; that check is one compare
    per length and always on. With `checked` the entry also refuses a null
    buffer that has elements and any two buffers whose byte ranges overlap,
    which is the disjointness the descriptor convention declares and the
    schedule transforms rely on; a valid call pays a handful of compares per
    buffer pair, so the mode is off by default. The record moves to the
    wrapper, where the header printer reads it.
  }];
  let options = [
    Option<"checked", "checked", "bool", /*default=*/"false",
           "Refuse a null buffer with elements and overlapping buffers before "
           "the kernel runs">,
  ];
  let constructor = "ondrix::createEmitOndrixCEntryPointsPass()";
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}

def UnrollOndspFpOrderedReduce
    : Pass<"unroll-ondsp-fp-ordered-reduce", "::mlir::ModuleOp"> {
  let summary = "Straight-line the f32 reductions whose contract already fixed their fold order";
  let description = [{
    Replaces two order-declared f32 shapes by the straight-line chain they
    denote: a rank-1 memref `ondsp.reduce_mac` under `contract = off` or
    `contract = fma` with static equal extents, and an `scf.for` carrying one
    f32 accumulator over a compile-time trip count, which is what the FIR
    family's tap lowering leaves behind. Both rewrites replay the same body at
    constant induction values, so every product and every fold keeps its
    operands, its position in the sequence, and its rounding; no numeric
    permission is involved and `contract = fast` is excluded, its sites
    carrying spent-permission records this pass must not replicate.

    The motive is a scalar target: the loop form pays an index update and a
    branch per term on a core with no lanes to amortise them, and a declared
    tap count is exactly the static structure that makes the loop avoidable.
    Nothing downstream chooses this shape otherwise -- the fast reduction pass
    refuses an ordered contract by construction, so before this pass an ordered
    f32 reduction reached the backend as a loop whose unrolling was left to a
    generic cost heuristic that counts a separate multiply and add as two.

    `vector-width` is the target's lane count at f32. Above one lane the
    reduction rewrite stands down and only the loop rewrite runs: the scalar
    lowering already gives an ordered contiguous `reduce_mac` a lane-blocked
    schedule, which reads the same terms in the same order and beats this
    chain, and claiming the site here would replace it with scalar loads
    (measured: 12% on a 64-term dot and 14% on rms at 128 bits, against 32% the
    other way on a matmul, whose accumulator loop no lane stage claims).

    `max-straight-line-terms` caps one reduction or loop; `max-unrolled-terms`
    bounds the terms a whole function may emit, because a lowering that already
    replicated its reductions multiplies a per-site bound by that replication.
    Past the budget every site in that function keeps its loop, so the shape
    stays uniform. Zero disables either bound.

    A loop that stores an output every trip is a sample recurrence, not an
    accumulator: replaying all its trips hands the backend one block of
    independent per-sample products that it hoists into register spills
    (measured on an in-order AArch64 model: 128 straight-lined f32 SOS samples
    ran 16% slower than the same chain in blocks of 16, and a four-section
    filter left as a loop ran 21% slower than in blocks of 4; 112-term blocks
    were the best of 28 to 224). Such a loop is replayed in blocks instead: the
    largest divisor of the trip count whose block holds at most
    `max-block-terms` multiplies and folds, one left as the loop.
  }];
  let options = [
    Option<"vectorWidth", "vector-width", "int64_t", /*default=*/"1",
           "Target lane count at f32; above one the reduction rewrite stands down">,
    Option<"maxStraightLineTerms", "max-straight-line-terms", "int64_t", /*default=*/"64",
           "Longest straight-line chain one site may become">,
    Option<"maxUnrolledTerms", "max-unrolled-terms", "int64_t", /*default=*/"0",
           "Per-function budget of straight-line terms; 0 leaves it unbounded">,
    Option<"maxBlockTerms", "max-block-terms", "int64_t", /*default=*/"128",
           "Terms one replayed block of a storing sample loop may hold; 0 keeps such loops">,
  ];
  let constructor = "ondrix::createUnrollOndspFpOrderedReducePass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::math::MathDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "ondrix::ondsp::OndspDialect",
  ];
}

def ScalarizeOndspFixedReduceMac
    : Pass<"scalarize-ondsp-fixed-reduce-mac", "::mlir::ModuleOp"> {
  let summary = "Expand ordered fixed reduce_mac into its definitional mac loop";
  let description = [{
    Expands each rank-1 memref `ondsp.reduce_mac` with a fixed numeric policy
    into the scf.for loop of explicit `ondsp.mac` updates that the
    operation's ordered left-fold contract denotes: one full declared update
    per index, in increasing index order, from the same initial accumulator.
    The expansion is exact by definition — it is the contract, not a
    reordering — and dynamic lengths get the same runtime equal-length proof
    the fixed scalar lowering inserts.

    The purpose is the target path: `convert-ondsp-to-ortumcore` admits
    explicit accumulator updates but deliberately rejects the fused
    reduction form, so a contract-form program runs this pass to reach
    target capability selection. Floating-point, vector-operand, and
    multi-lane reductions are left untouched for their own consumers, and
    anything this pass does not expand still fails closed downstream.

    `max-unrolled-terms` bounds the straight-line form by the terms a whole
    function would emit, not by one reduction's length: a lowering that has
    already replicated its reductions (one per output row, as the DCT does)
    multiplies a per-reduction bound by that replication and buys instruction
    count with instruction cache. Past the budget every reduction in that
    function keeps the loop form, so the shape stays uniform. Zero disables
    the budget, which is what a target selecting the capability wants. The
    canonical pipeline's budget of 128 terms is a host-class measurement: at
    32 terms a filter tap chain runs 1.9x faster for 2.7x its bytes, while
    the DCT's per-row replication reaches 4096 terms and 15x its bytes to run
    slower.
  }];
  let options = [
    Option<"maxUnrolledTerms", "max-unrolled-terms", "int64_t", /*default=*/"0",
           "Per-function budget of straight-line reduction terms; 0 leaves the "
           "straight-line form unbounded">,
  ];
  let constructor = "ondrix::createScalarizeOndspFixedReduceMacPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::cf::ControlFlowDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "ondrix::ondsp::OndspDialect",
  ];
}

def ScalarizeOndspCertifiedConstantReduce
    : Pass<"scalarize-ondsp-certified-constant-reduce", "::mlir::ModuleOp"> {
  let summary = "Expand a constant-coefficient fixed reduce_mac into certified 32-bit term "
                "groups on a wrapping accumulator";
  let description = [{
    Rewrites a zero-seeded rank-1 memref `ondsp.reduce_mac` whose coefficient
    operand is a constant global (or a static reversed view of one) into
    scalar arithmetic the prefix-range analysis has certified: every exact
    product is formed in the 32-bit term width, consecutive products are
    summed there in groups of the widest certified power-of-two width, and
    each group sum enters a WRAPPING accumulator of the declared storage
    through `ondsp.acc_add_term`. The definitional form the sibling
    `scalarize-ondsp-fixed-reduce-mac` emits updates the accumulator once per
    product under the declared overflow mode, which on a target without the
    accumulator's width is a widened multiply and a rail check per term.

    Two certificates authorize the rewrite, both rebuilt from the constants
    and the complete signed input domain. A saturating accumulator is admitted
    only when `proveOrderedZeroSeededConstantReduction` shows that no ordered
    prefix can reach its rails: the saturating and the wrapping update
    sequences then denote the same exact sum, so the accumulator is retyped
    to wrap and the rail checks are gone. A wrapping accumulator needs no rail
    certificate; its update is the exact-modulo class. The group width is
    `largestCertifiedTermGroup` at 32 bits: the interval sum of each aligned
    group of that width fits a signed 32-bit term, and since two's-complement
    addition is associative a group whose total fits is exact in whatever
    order its 32-bit adds run. Products past the last full group are added
    singly, which is what the certificate covers for them. A group width the
    analysis cannot certify leaves every product a single term, still exact
    and still cheaper than the widened multiply.

    The group width is `scalar-register-bits`, and it is the whole reason the
    rewrite is worth anything. What a group buys is ONE carrier-width add
    where the declared form takes one per product, and a carrier-width add is
    only expensive on a machine whose registers are narrower than the carrier:
    on a 32-bit target it is an add, a carry test and a second add, and on a
    64-bit one it is a single instruction. So the group sums as narrow as the
    exact product allows but never narrower than one register, and where the
    registers already hold the carrier the narrowing is skipped entirely --
    there the only thing left to win is the per-term loop overhead a block
    amortizes, so a reduction the budget already straight-lined is left to the
    definitional expansion, which is then the better form.

    Straight-line or loop form follows the sibling pass's rule and budgets: a
    reduction of at most 64 terms unrolls with its coefficients as constants
    unless the function's straight-line total exceeds `max-unrolled-terms`;
    otherwise a loop steps over full groups of at most 16 terms, loading the
    coefficients, and the remaining products follow straight-line. Runtime
    coefficients, a product with a shift, a multi-lane or unsigned
    accumulator, a result read by anything but `acc_export`, dynamic or
    unequal extents, and coefficient tables longer than `max-elements` are
    left byte-identical for the definitional expansion.

    Measured on the canonical pipeline at width zero, RV32IM under gem5,
    instructions per call: the definitional expansion ran the 64-term
    constant dot in 593, the 16-tap filter in 6761, the DCT-8 in 676, the
    DCT-32 in 28761 and the DCT-64 in 112797; the certified groups run them
    in 468, 5242, 506, 6584 and 36798. The loop-form rows gain most, because
    their saturating step was a clamp pair per product; the DCT-8 still trails
    the tensor lowering's 447, whose straight-line i64 products the middle
    end shares across rows.
  }];
  let options = [
    Option<"maxUnrolledTerms", "max-unrolled-terms", "int64_t", /*default=*/"0",
           "Per-function budget of straight-line reduction terms shared with "
           "scalarize-ondsp-fixed-reduce-mac; 0 leaves the straight-line form unbounded">,
    Option<"maxElements", "max-elements", "int64_t", /*default=*/"256",
           "Longest constant coefficient table the certificates are built for">,
    Option<"scalarRegisterBits", "scalar-register-bits", "int64_t", /*default=*/"32",
           "Declared target quantity: the machine's integer register width, which "
           "decides both the width a group sums in and whether a straight-lined "
           "reduction is worth rewriting at all">,
  ];
  let constructor = "ondrix::createScalarizeOndspCertifiedConstantReducePass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "ondrix::ondsp::OndspDialect",
  ];
}

def UnrollOndspFixedMacLoops
    : Pass<"unroll-ondsp-fixed-mac-loops", "::mlir::ModuleOp"> {
  let summary = "Unroll short static accumulator-carried mac loops into straight-line chains";
  let description = [{
    Fully unrolls an `scf.for` whose single iteration argument is a
    single-lane fixed accumulator advanced through `ondsp.mac`, when the
    trip count is a compile-time constant of at most 64. Unrolling is the
    identity on the declared update sequence — every iteration body executes
    unchanged, in the same order, at a constant induction value — so no
    numeric proof is involved. The motive is the target path shared with
    `scalarize-ondsp-fixed-reduce-mac`'s straight-line form: the loop form
    pays an index update and a branch per term, which outweighs the term
    itself on the MAC capability, and 64 is that same measured bound. This
    reaches the accumulator loops that never pass through `reduce_mac`,
    such as the tensor-form FIR lowering's tap loop.

    Everything else keeps its form and every downstream consumer's
    legality: dynamic or non-positive bounds, trip counts past the bound,
    more than one iteration argument, multi-lane accumulators, and loops
    whose body contains no mac. A static extent read through `tensor.dim`
    counts as a compile-time constant, which is how the tap loop presents
    its bound before any folding.

    `max-unrolled-terms` bounds the iterations a whole function may unroll,
    the same budget and for the same reason as
    `scalarize-ondsp-fixed-reduce-mac`: a lowering that already replicated
    its loops multiplies a per-loop bound by that replication. The two
    budgets must agree, or this pass re-expands what the other declined.
    Zero disables the budget.
  }];
  let options = [
    Option<"maxUnrolledTerms", "max-unrolled-terms", "int64_t", /*default=*/"0",
           "Per-function budget of unrolled iterations; 0 leaves unrolling "
           "unbounded">,
  ];
  let constructor = "ondrix::createUnrollOndspFixedMacLoopsPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
    "ondrix::ondsp::OndspDialect",
  ];
}

def RelaxOndspUnreachableSaturation
    : Pass<"relax-ondsp-unreachable-saturation", "::mlir::ModuleOp"> {
  let summary = "Drop a declared saturating boundary a range proof shows can never fire";
  let description = [{
    Rewrites an `ondsp.round_shift` whose declared overflow is `saturate` to
    `wrap` when the pass proves the shifted reading always lies inside the
    destination storage. The two declarations denote the same function there,
    so the rewrite is bit-exact by construction and the emitted code loses the
    clamp: on rv32 an i64 reading narrowed to i16 costs fourteen instructions
    for a comparison that the proof settles once, at compile time.

    The bound is a signed interval walked back from the shift. A leaf, and any
    producer outside the understood set, contributes its own type's full signed
    range, so the walk is total rather than fail-closed; what keeps it sound is
    that every arithmetic step whose exact interval would leave its own result
    type falls back to that range as well, which is exactly what a wrapping
    operation may produce. The understood set is the constants, sign extension,
    truncation, addition, subtraction and multiplication the fixed routes emit,
    plus the accumulator web: `acc_zero`, `acc_add_term` under its declared
    update overflow, `acc_export`, and a chained `round_shift`.

    This is where a declared output format earns its headroom. A transform
    whose output frac leaves the reading a bit of room -- the DCT family, whose
    k = 0 row is the extreme case -- never reaches its rails, and the check is
    then pure cost; a filter whose taps can drive the accumulator past the
    destination keeps every clamp it declared.

    The rewrite weakens a policy the ondsp-to-ortumcore conversion may require,
    so it belongs to the generic lowering flow: a target route that needs the
    saturating readout runs its own pass list and never sees it.
  }];
  let constructor = "ondrix::createRelaxOndspUnreachableSaturationPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
    "ondrix::ondsp::OndspDialect",
  ];
}

def WidenOndspExactAccumulators
    : Pass<"widen-ondsp-exact-accumulators", "::mlir::ModuleOp"> {
  let summary = "Retype provably exact wrap accumulators to the saturating target domain";
  let description = [{
    Retypes signed frac-30 wrapping accumulators of width at most 40 to the
    i40 saturating target domain when the pass itself re-proves that no
    update can overflow. Nothing is trusted: the proof is rebuilt from the
    static structure of the accumulator web, so a hand-written web earns the
    same widening the frontend's inferred default does.

    The provable shape is one acc_zero flowing through full signed-Q15 macs
    — individually, as the iteration argument of scf.for loops with
    compile-time trip counts, or as one ordered reduce_mac whose rank-1
    element count is static — into one acc_export. Every full Q15 product is
    bounded by
    2^30 in magnitude, so every partial sum is bounded by U * 2^30 for U the
    largest possible number of updates; when that bound stays below
    2^(W-1) - 1 the declared wrap provably never fires, the same bound keeps
    the 40-bit saturation unreachable, and both declarations denote the exact
    sum. Any escape from the shape (multiple uses, a dynamic trip count, a
    non-Q15 product, acc_import, more than one lane) leaves the web
    untouched; the downstream ondsp-to-ortumcore conversion then fails closed
    exactly as before, so this pass can only widen, never weaken.
  }];
  let constructor = "ondrix::createWidenOndspExactAccumulatorsPass()";
  let dependentDialects = [
    "mlir::scf::SCFDialect",
    "ondrix::ondsp::OndspDialect",
  ];
}

#endif // ONDRIX_TRANSFORMS_PASSES

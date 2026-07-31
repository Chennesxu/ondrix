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
    next stage reads it back with constant-index `tensor.extract`. Upstream
    folding only inspects the nearest insert, so a chain whose last write
    targets a different index leaves the whole intermediate materialized.

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

#endif // ONDRIX_TRANSFORMS_PASSES

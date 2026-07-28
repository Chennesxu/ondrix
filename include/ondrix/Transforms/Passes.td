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

#endif // ONDRIX_TRANSFORMS_PASSES

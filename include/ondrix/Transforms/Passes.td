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

#endif // ONDRIX_TRANSFORMS_PASSES

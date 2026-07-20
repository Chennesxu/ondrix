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
  let summary = "Tile valid FIR filters along the output axis";
  let description = [{
    Opt-in output-axis tiling for tensor `ondrix.fir_filter` operations. The
    pass delegates loop construction and destination insertion to MLIR's SCF
    TilingInterface implementation. Each output tile reads the corresponding
    input halo and retains the complete coefficient tensor; the ordered tap
    reduction remains inside the tiled operation and is not reassociated.
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

#endif // ONDRIX_TRANSFORMS_PASSES

#include "ondrix/Transforms/Passes.h"

#include "ondrix/Dialect/ondrix/IR/OndrixDialect.h"
#include "ondrix/Dialect/ondrix/IR/OndrixOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"

namespace ondrix {
#define GEN_PASS_DEF_TILEONDRIXFIRFILTER
#include "ondrix/Transforms/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

void assertValidFirFilterShape(ondrix::ir::FirFilterOp op, OpBuilder &builder) {
  Location loc = op.getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value inputLength = builder.create<tensor::DimOp>(loc, op.getInput(), zero);
  Value coefficientLength = builder.create<tensor::DimOp>(loc, op.getCoeffs(), zero);
  Value outputLength = builder.create<tensor::DimOp>(loc, op.getInit(), zero);

  Value hasCoefficients =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, coefficientLength, zero);
  builder.create<cf::AssertOp>(
      loc, hasCoefficients, builder.getStringAttr("valid FIR requires at least one coefficient"));
  Value inputCoversWindow =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge, inputLength, coefficientLength);
  builder.create<cf::AssertOp>(
      loc, inputCoversWindow,
      builder.getStringAttr("valid FIR input must cover one coefficient window"));
  Value remaining = builder.create<arith::SubIOp>(loc, inputLength, coefficientLength);
  Value requiredOutputLength = builder.create<arith::AddIOp>(loc, remaining, one);
  Value outputMatches = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, outputLength,
                                                      requiredOutputLength);
  builder.create<cf::AssertOp>(
      loc, outputMatches,
      builder.getStringAttr("valid FIR output length must equal input length minus coefficient "
                            "length plus one"));
}

class TileOndrixFirFilterPass final
    : public ondrix::impl::TileOndrixFirFilterBase<TileOndrixFirFilterPass> {
public:
  using ondrix::impl::TileOndrixFirFilterBase<TileOndrixFirFilterPass>::TileOndrixFirFilterBase;

  void runOnOperation() override {
    if (tileSize <= 0) {
      getOperation().emitError("tile-size must be positive");
      signalPassFailure();
      return;
    }

    SmallVector<ondrix::ir::FirFilterOp> filters;
    getOperation().walk([&](ondrix::ir::FirFilterOp op) { filters.push_back(op); });

    IRRewriter rewriter(&getContext());
    scf::SCFTilingOptions options;
    SmallVector<int64_t> tileSizes{tileSize};
    options.setTileSizes(tileSizes);
    for (ondrix::ir::FirFilterOp filter : filters) {
      rewriter.setInsertionPoint(filter);
      assertValidFirFilterShape(filter, rewriter);
      FailureOr<scf::SCFTilingResult> tiled =
          scf::tileUsingSCFForOp(rewriter, cast<TilingInterface>(filter.getOperation()), options);
      if (failed(tiled)) {
        filter.emitOpError("failed output-axis SCF tiling");
        signalPassFailure();
        return;
      }
      rewriter.replaceOp(filter, tiled->replacements);
    }
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createTileOndrixFirFilterPass() {
  return std::make_unique<TileOndrixFirFilterPass>();
}

std::unique_ptr<Pass>
ondrix::createTileOndrixFirFilterPass(const TileOndrixFirFilterOptions &options) {
  return std::make_unique<TileOndrixFirFilterPass>(options);
}

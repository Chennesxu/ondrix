#include "ondrix/Transforms/Passes.h"

#include "ondrix/Analysis/ConstantSequenceAnalysis.h"
#include "ondrix/Dialect/ondrix/IR/OndrixDialect.h"
#include "ondrix/Dialect/ondrix/IR/OndrixOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <limits>

namespace ondrix {
#define GEN_PASS_DEF_SPECIALIZEONDRIXCONSTANTFIR
#include "ondrix/Transforms/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

FailureOr<ondrix::ConstantSequenceFacts> getConstantCoefficientFacts(ondrix::ir::FirOp op,
                                                                     int64_t maxTaps) {
  auto getGlobal = op.getCoeffs().getDefiningOp<memref::GetGlobalOp>();
  if (!getGlobal)
    return failure();

  auto global =
      SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(getGlobal, getGlobal.getNameAttr());
  if (!global)
    return failure();

  auto initializer = dyn_cast_or_null<DenseIntElementsAttr>(global.getConstantInitValue());
  if (!initializer || initializer.getType().getRank() != 1)
    return failure();

  return ondrix::analyzeConstantIntegerSequence(initializer, maxTaps);
}

Value createIndex(Location loc, int64_t index, PatternRewriter &rewriter) {
  return rewriter.create<arith::ConstantIndexOp>(loc, index);
}

Value createIntegerConstant(Location loc, IntegerType type, const llvm::APInt &value,
                            PatternRewriter &rewriter) {
  return rewriter.create<arith::ConstantOp>(loc, type, IntegerAttr::get(type, value));
}

Value createInputLoad(ondrix::ir::FirOp op, int64_t index, PatternRewriter &rewriter) {
  Value position = createIndex(op.getLoc(), index, rewriter);
  return rewriter.create<memref::LoadOp>(op.getLoc(), op.getInput(), position);
}

Value createMacUpdate(ondrix::ir::FirOp op, Value accumulator, int64_t index,
                      const llvm::APInt &coefficient, ondrix::ondsp::FixedAttr numeric,
                      PatternRewriter &rewriter) {
  auto storage = cast<IntegerType>(numeric.getStorage());
  Value input = createInputLoad(op, index, rewriter);
  Value constant = createIntegerConstant(op.getLoc(), storage, coefficient, rewriter);
  return rewriter.create<ondrix::ondsp::MacOp>(op.getLoc(), accumulator.getType(), accumulator,
                                               input, constant, numeric, *op.getProduct());
}

Value createSymmetricPairUpdate(ondrix::ir::FirOp op, Value accumulator, int64_t lhsIndex,
                                int64_t rhsIndex, const llvm::APInt &coefficient,
                                ondrix::ondsp::FixedAttr numeric,
                                const ondrix::ondsp::ProductSemantics &productSemantics,
                                PatternRewriter &rewriter) {
  auto storage = cast<IntegerType>(numeric.getStorage());
  unsigned storageWidth = storage.getWidth();

  // The widened pre-add and product preserve the mathematical distributive
  // identity before the wrapping accumulator applies its modulo boundary.
  IntegerType sumType = rewriter.getIntegerType(storageWidth + 1);
  IntegerType termType = rewriter.getIntegerType(productSemantics.rawWidth + 1);
  Value lhs = createInputLoad(op, lhsIndex, rewriter);
  Value rhs = createInputLoad(op, rhsIndex, rewriter);
  Value lhsExtended = rewriter.create<arith::ExtSIOp>(op.getLoc(), sumType, lhs);
  Value rhsExtended = rewriter.create<arith::ExtSIOp>(op.getLoc(), sumType, rhs);
  Value sum = rewriter.create<arith::AddIOp>(op.getLoc(), lhsExtended, rhsExtended);
  Value widenedSum = rewriter.create<arith::ExtSIOp>(op.getLoc(), termType, sum);

  Value constant = createIntegerConstant(op.getLoc(), storage, coefficient, rewriter);
  Value widenedConstant = rewriter.create<arith::ExtSIOp>(op.getLoc(), termType, constant);
  Value term = rewriter.create<arith::MulIOp>(op.getLoc(), widenedSum, widenedConstant);
  auto termNumeric = ondrix::ondsp::FixedAttr::get(rewriter.getContext(), numeric.getSignedness(),
                                                   termType, productSemantics.frac);
  return rewriter.create<ondrix::ondsp::AccAddTermOp>(op.getLoc(), accumulator.getType(),
                                                      accumulator, term, termNumeric);
}

class SpecializeConstantFirPattern final : public OpRewritePattern<ondrix::ir::FirOp> {
public:
  SpecializeConstantFirPattern(MLIRContext *context, int64_t maxTaps)
      : OpRewritePattern(context), maxTaps(maxTaps) {}

  LogicalResult matchAndRewrite(ondrix::ir::FirOp op, PatternRewriter &rewriter) const override {
    auto numeric = dyn_cast<ondrix::ondsp::FixedAttr>(op.getNumeric());
    auto accumulator = dyn_cast<ondrix::ondsp::AccType>(op.getResult().getType());
    auto inputType = dyn_cast<MemRefType>(op.getInput().getType());
    auto coefficientType = dyn_cast<MemRefType>(op.getCoeffs().getType());
    if (!numeric || !accumulator || !op.getProduct() || !inputType || !coefficientType ||
        inputType.getRank() != 1 || coefficientType.getRank() != 1 || inputType.isDynamicDim(0) ||
        coefficientType.isDynamicDim(0))
      return failure();

    FailureOr<ondrix::ConstantSequenceFacts> facts = getConstantCoefficientFacts(op, maxTaps);
    if (failed(facts) || facts->getElementCount() != inputType.getDimSize(0))
      return failure();

    FailureOr<ondrix::ondsp::ProductSemantics> productSemantics =
        ondrix::ondsp::inferProductSemantics(op, numeric, *op.getProduct());
    if (failed(productSemantics))
      return failure();

    bool canEliminateZeroTaps =
        facts->hasZero() && ondrix::ondsp::classifyZeroProductElimination(numeric) ==
                                ondrix::ondsp::TransformEquivalence::BitExact;
    bool canPairSymmetricTaps = facts->getElementCount() >= 2 && facts->isSymmetric() &&
                                productSemantics->rawWidth < std::numeric_limits<unsigned>::max() &&
                                ondrix::ondsp::classifyDistributiveProductPairing(
                                    numeric, *productSemantics, accumulator) ==
                                    ondrix::ondsp::TransformEquivalence::ExactModulo;
    if (!canEliminateZeroTaps && !canPairSymmetricTaps)
      return failure();

    Value current =
        rewriter.create<ondrix::ondsp::AccZeroOp>(op.getLoc(), op.getResult().getType());
    if (canPairSymmetricTaps) {
      llvm::ArrayRef<llvm::APInt> values = facts->getValues();
      size_t length = values.size();
      for (size_t index = 0; index < length / 2; ++index) {
        const llvm::APInt &coefficient = values[index];
        if (coefficient.isZero())
          continue;
        current = createSymmetricPairUpdate(op, current, index, length - index - 1, coefficient,
                                            numeric, *productSemantics, rewriter);
      }
      if (length % 2 != 0) {
        size_t center = length / 2;
        if (!values[center].isZero())
          current = createMacUpdate(op, current, center, values[center], numeric, rewriter);
      }
    } else {
      for (const auto &[index, coefficient] : llvm::enumerate(facts->getValues())) {
        if (coefficient.isZero())
          continue;
        current = createMacUpdate(op, current, index, coefficient, numeric, rewriter);
      }
    }

    auto getGlobal = op.getCoeffs().getDefiningOp<memref::GetGlobalOp>();
    rewriter.replaceOp(op, current);
    if (getGlobal && getGlobal->use_empty())
      rewriter.eraseOp(getGlobal);
    return success();
  }

private:
  int64_t maxTaps;
};

class SpecializeOndrixConstantFirPass final
    : public ondrix::impl::SpecializeOndrixConstantFirBase<SpecializeOndrixConstantFirPass> {
public:
  using ondrix::impl::SpecializeOndrixConstantFirBase<
      SpecializeOndrixConstantFirPass>::SpecializeOndrixConstantFirBase;

  void runOnOperation() override {
    if (maxTaps <= 0) {
      getOperation().emitError("max-taps must be positive");
      signalPassFailure();
      return;
    }
    RewritePatternSet patterns(&getContext());
    patterns.add<SpecializeConstantFirPattern>(&getContext(), maxTaps);
    SmallVector<Operation *> firOperations;
    getOperation().walk(
        [&](ondrix::ir::FirOp operation) { firOperations.push_back(operation.getOperation()); });
    if (firOperations.empty())
      return;

    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyOpPatternsAndFold(firOperations, frozenPatterns, config)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createSpecializeOndrixConstantFirPass() {
  return std::make_unique<SpecializeOndrixConstantFirPass>();
}

std::unique_ptr<Pass>
ondrix::createSpecializeOndrixConstantFirPass(const SpecializeOndrixConstantFirOptions &options) {
  return std::make_unique<SpecializeOndrixConstantFirPass>(options);
}

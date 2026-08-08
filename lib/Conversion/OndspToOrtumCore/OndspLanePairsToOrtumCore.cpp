#include "OrtumCoreLoweringSupport.h"
#include "ondrix/Conversion/OndspToOrtumCore/OndspToOrtumCore.h"
#include "ondrix/Conversion/Utils/ConversionLegality.h"

#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreDialect.h"
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreOps.h"
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/OneToNTypeConversion.h"

namespace ondrix {
#define GEN_PASS_DEF_CONVERTONDSPLANEPAIRSTOORTUMCORE
#include "ondrix/Conversion/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

static bool isPairAccumulator(Type type) {
  auto accumulator = dyn_cast<ondrix::ondsp::AccType>(type);
  return accumulator && accumulator.getLanes() == 2 &&
         ondrix::conversion::isOrtumCoreLaneDomain(accumulator);
}

// The admitted export policy when its vector result matches, else nullopt.
static std::optional<ondrix::conversion::OrtumCoreExportPolicy>
classifyPairExport(ondrix::ondsp::AccExportOp op) {
  std::optional<ondrix::conversion::OrtumCoreExportPolicy> policy =
      ondrix::conversion::classifyOrtumCoreExport(op);
  auto result = dyn_cast<VectorType>(op.getResult().getType());
  if (!policy || !result || result.getNumElements() != 2 ||
      result.getElementType() != policy->storage)
    return std::nullopt;
  return policy;
}

static bool touchesPairAccumulator(Operation *op) {
  auto matches = [](TypeRange types) {
    return ondrix::conversion::containsMatchingType(types, isPairAccumulator);
  };
  if (matches(op->getOperandTypes()) || matches(op->getResultTypes()))
    return true;
  for (Region &region : op->getRegions())
    for (Block &block : region)
      if (matches(block.getArgumentTypes()))
        return true;
  return false;
}

// The 1:N driver hard-fails on boundary casts it cannot resolve, so a
// function is rewritten only after every operation touching a pair value is
// proven convertible; rejected functions reach convert-ondsp-to-ortumcore
// unchanged and fail closed there.
static bool isConvertibleFunction(func::FuncOp function) {
  FunctionType signature = function.getFunctionType();
  if (ondrix::conversion::containsMatchingType(signature.getInputs(), isPairAccumulator) ||
      ondrix::conversion::containsMatchingType(signature.getResults(), isPairAccumulator))
    return false;
  WalkResult result = function.walk([](Operation *op) {
    if (isa<func::FuncOp>(op) || !touchesPairAccumulator(op))
      return WalkResult::advance();
    if (isa<ondrix::ondsp::AccZeroOp, scf::ForOp, scf::YieldOp, scf::IfOp, scf::WhileOp,
            scf::ConditionOp>(op))
      return WalkResult::advance();
    if (auto mac = dyn_cast<ondrix::ondsp::MacOp>(op))
      return ondrix::conversion::isOrtumCoreMacPolicy(mac) ? WalkResult::advance()
                                                           : WalkResult::interrupt();
    if (auto exported = dyn_cast<ondrix::ondsp::AccExportOp>(op))
      return classifyPairExport(exported) ? WalkResult::advance() : WalkResult::interrupt();
    return WalkResult::interrupt();
  });
  return !result.wasInterrupted();
}

class PairTypeConverter final : public OneToNTypeConverter {
public:
  explicit PairTypeConverter(MLIRContext *context) {
    addConversion([](Type type) { return type; });
    addConversion([context](ondrix::ondsp::AccType type,
                            SmallVectorImpl<Type> &results) -> std::optional<LogicalResult> {
      if (!isPairAccumulator(type))
        return std::nullopt;
      results.append(2, ondrix::ortumcore::AccumType::get(context));
      return success();
    });
  }
};

class PairAccZeroLowering final : public OneToNOpConversionPattern<ondrix::ondsp::AccZeroOp> {
public:
  using OneToNOpConversionPattern<ondrix::ondsp::AccZeroOp>::OneToNOpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ondsp::AccZeroOp op, OpAdaptor adaptor,
                                OneToNPatternRewriter &rewriter) const override {
    const OneToNTypeMapping &mapping = adaptor.getResultMapping();
    if (!mapping.hasNonIdentityConversion())
      return failure();
    SmallVector<Value> lanes;
    for (Type type : mapping.getConvertedTypes(0))
      lanes.push_back(rewriter.create<ondrix::ortumcore::AccInitOp>(op.getLoc(), type));
    rewriter.replaceOp(op, lanes, mapping);
    return success();
  }
};

class PairMacLowering final : public OneToNOpConversionPattern<ondrix::ondsp::MacOp> {
public:
  using OneToNOpConversionPattern<ondrix::ondsp::MacOp>::OneToNOpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ondsp::MacOp op, OpAdaptor adaptor,
                                OneToNPatternRewriter &rewriter) const override {
    if (!isPairAccumulator(op.getAcc().getType()) || !ondrix::conversion::isOrtumCoreMacPolicy(op))
      return failure();
    ValueRange lanes = adaptor.getAcc();
    if (lanes.size() != 2)
      return failure();

    Location loc = op.getLoc();
    Value values = adaptor.getLhs().front();
    Value coefficient = adaptor.getRhs().front();
    Value value0 = rewriter.create<vector::ExtractOp>(loc, values, 0);
    Value value1 = rewriter.create<vector::ExtractOp>(loc, values, 1);
    // The declared broadcast coefficient becomes both dual-lane multiplier
    // halves; lane independence is the dmac contract.
    auto dmac = rewriter.create<ondrix::ortumcore::DmacOp>(
        loc, lanes[0].getType(), lanes[1].getType(), lanes[0], lanes[1], value0, coefficient,
        value1, coefficient);
    rewriter.replaceOp(op, dmac->getResults(), adaptor.getResultMapping());
    return success();
  }
};

class PairAccExportLowering final : public OneToNOpConversionPattern<ondrix::ondsp::AccExportOp> {
public:
  using OneToNOpConversionPattern<ondrix::ondsp::AccExportOp>::OneToNOpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ondsp::AccExportOp op, OpAdaptor adaptor,
                                OneToNPatternRewriter &rewriter) const override {
    std::optional<ondrix::conversion::OrtumCoreExportPolicy> policy = classifyPairExport(op);
    if (!isPairAccumulator(op.getAcc().getType()) || !policy)
      return failure();
    ValueRange lanes = adaptor.getAcc();
    if (lanes.size() != 2)
      return failure();

    Location loc = op.getLoc();
    auto resultType = cast<VectorType>(op.getResult().getType());
    Value result = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(resultType));
    for (auto [index, lane] : llvm::enumerate(lanes)) {
      Value exported = ondrix::conversion::emitOrtumCoreReadout(rewriter, loc, lane, *policy);
      result =
          rewriter.create<vector::InsertOp>(loc, exported, result, static_cast<int64_t>(index));
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

class ConvertOndspLanePairsToOrtumCorePass final
    : public ondrix::impl::ConvertOndspLanePairsToOrtumCoreBase<
          ConvertOndspLanePairsToOrtumCorePass> {
public:
  using ondrix::impl::ConvertOndspLanePairsToOrtumCoreBase<
      ConvertOndspLanePairsToOrtumCorePass>::ConvertOndspLanePairsToOrtumCoreBase;

  void runOnOperation() override {
    for (func::FuncOp function : getOperation().getOps<func::FuncOp>()) {
      if (function.isExternal() || !isConvertibleFunction(function))
        continue;
      PairTypeConverter typeConverter(&getContext());
      RewritePatternSet patterns(&getContext());
      patterns.add<PairAccZeroLowering, PairMacLowering, PairAccExportLowering>(typeConverter,
                                                                                &getContext());
      scf::populateSCFStructuralOneToNTypeConversions(typeConverter, patterns);
      if (failed(applyPartialOneToNConversion(function, typeConverter, std::move(patterns))))
        return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createConvertOndspLanePairsToOrtumCorePass() {
  return std::make_unique<ConvertOndspLanePairsToOrtumCorePass>();
}

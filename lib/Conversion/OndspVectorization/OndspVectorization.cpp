#include "ondrix/Conversion/OndspVectorization/OndspVectorization.h"
#include "ondrix/Conversion/Utils/FixedPointDomainUtils.h"
#include "ondrix/Conversion/Utils/FixedPointVectorUtils.h"
#include "ondrix/Conversion/Utils/ReductionUtils.h"

#include "ondrix/Analysis/ConstantSequenceAnalysis.h"
#include "ondrix/Analysis/FixedPointPrefixRangeAnalysis.h"
#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <cassert>
#include <limits>

namespace ondrix {
#define GEN_PASS_DEF_VECTORIZEONDSPCONSTANTSATURATINGMEMREFREDUCE
#define GEN_PASS_DEF_VECTORIZEONDSPFIXEDMEMREFREDUCE
#define GEN_PASS_DEF_VERIFYONDSPCONSTANTREASSOCIATIONPROOFTRACE
#include "ondrix/Conversion/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

constexpr uint64_t maxProofTraceBytes = 64ULL * 1024 * 1024;
constexpr int64_t maxProofTraceElements = 65536;
constexpr unsigned maxProofTraceAPIntWidth = 4096;

bool isRepresentableLLVMAddressSpace(IntegerAttr memorySpace) {
  const llvm::APInt &value = memorySpace.getValue();
  return !value.isNegative() && value.getActiveBits() <= std::numeric_limits<unsigned>::digits;
}

bool hasDefaultLLVMVectorMemorySpace(MemRefType type) {
  Attribute memorySpace = type.getMemorySpace();
  if (!memorySpace)
    return true;
  auto integerSpace = dyn_cast<IntegerAttr>(memorySpace);
  return integerSpace && integerSpace.getValue().isZero();
}

bool hasInvalidLLVMIntegerMemorySpace(MemRefType type) {
  auto memorySpace = dyn_cast_or_null<IntegerAttr>(type.getMemorySpace());
  return memorySpace && !isRepresentableLLVMAddressSpace(memorySpace);
}

bool hasInvalidLLVMIntegerMemorySpace(ondrix::ondsp::ReduceMacOp op) {
  auto lhsType = dyn_cast<MemRefType>(op.getLhs().getType());
  auto rhsType = dyn_cast<MemRefType>(op.getRhs().getType());
  return (lhsType && hasInvalidLLVMIntegerMemorySpace(lhsType)) ||
         (rhsType && hasInvalidLLVMIntegerMemorySpace(rhsType));
}

bool isSupportedMemRefReduction(ondrix::ondsp::ReduceMacOp op) {
  auto accumulator = dyn_cast<ondrix::ondsp::AccType>(op.getInitial().getType());
  auto numeric = dyn_cast<ondrix::ondsp::FixedAttr>(op.getNumeric());
  auto lhsType = dyn_cast<MemRefType>(op.getLhs().getType());
  auto rhsType = dyn_cast<MemRefType>(op.getRhs().getType());
  return accumulator && numeric && op.getProduct() && lhsType && rhsType &&
         lhsType.getRank() == 1 && rhsType.getRank() == 1 &&
         lhsType.getElementType() == numeric.getStorage() &&
         rhsType.getElementType() == numeric.getStorage() &&
         hasDefaultLLVMVectorMemorySpace(lhsType) && hasDefaultLLVMVectorMemorySpace(rhsType) &&
         isLastMemrefDimUnitStride(lhsType) && isLastMemrefDimUnitStride(rhsType) &&
         ondrix::conversion::isSupportedFixedVectorMacDomain(accumulator, numeric,
                                                             *op.getProduct());
}

Value createHorizontalAccumulatorUpdate(ondrix::ondsp::ReduceMacOp op, Value accumulator, Value lhs,
                                        Value rhs, OpBuilder &builder) {
  auto accumulatorType = cast<ondrix::ondsp::AccType>(accumulator.getType());
  auto numeric = cast<ondrix::ondsp::FixedAttr>(op.getNumeric());
  FailureOr<ondrix::conversion::FixedVectorProductTerms> terms =
      ondrix::conversion::lowerFixedVectorProductTerms(op, accumulatorType, numeric,
                                                       *op.getProduct(), lhs, rhs, builder);
  assert(succeeded(terms) && "validated fixed Vector product domain must lower");
  FailureOr<ondrix::conversion::FixedVectorHorizontalSum> horizontal =
      ondrix::conversion::lowerFixedVectorHorizontalSum(op, *terms, builder);
  assert(succeeded(horizontal) && "validated fixed Vector horizontal sum must lower");
  return builder.create<ondrix::ondsp::AccAddTermOp>(
      op.getLoc(), accumulator.getType(), accumulator, horizontal->sum, horizontal->numeric);
}

FailureOr<ondrix::analysis::NoOverflowChunkReassociationPlan>
planConstantSaturatingReduction(ondrix::ondsp::ReduceMacOp op, int64_t vectorWidth,
                                int64_t maxElements) {
  if (!isSupportedMemRefReduction(op) || !op.getInitial().getDefiningOp<ondrix::ondsp::AccZeroOp>())
    return failure();

  auto accumulator = cast<ondrix::ondsp::AccType>(op.getInitial().getType());
  auto numeric = cast<ondrix::ondsp::FixedAttr>(op.getNumeric());
  if (accumulator.getUpdateOverflow() != ondrix::ondsp::OverflowMode::Saturate ||
      !ondrix::ondsp::isFullProduct(*op.getProduct()))
    return failure();
  FailureOr<ondrix::conversion::SupportedFixedMacDomain> domain =
      ondrix::conversion::getSupportedFixedVectorMacDomain(op, accumulator, numeric,
                                                           *op.getProduct());
  if (failed(domain) || domain->termStorage.getWidth() > 64)
    return failure();

  FailureOr<ondrix::ConstantIntegerMemRefFacts> constant =
      ondrix::analyzeConstantIntegerMemRef(op.getRhs(), maxElements);
  if (failed(constant))
    return failure();
  auto rhsType = cast<MemRefType>(op.getRhs().getType());
  if (!rhsType.isDynamicDim(0) &&
      constant->getSequence().getElementCount() != rhsType.getDimSize(0))
    return failure();
  return ondrix::analysis::FixedPointPrefixRangePlanner::planZeroSeededConstantChunkReduction(
      op, *constant, vectorWidth);
}

class ConstantSaturatingReduceMacVectorization final
    : public OpRewritePattern<ondrix::ondsp::ReduceMacOp> {
public:
  ConstantSaturatingReduceMacVectorization(
      MLIRContext *context, int64_t vectorWidth, int64_t maxElements,
      const DenseMap<Operation *, int64_t> &subjectOrdinals,
      SmallVectorImpl<ondrix::analysis::NoOverflowChunkReassociationTrace> &proofTraces)
      : OpRewritePattern(context), vectorWidth(vectorWidth), maxElements(maxElements),
        subjectOrdinals(subjectOrdinals), proofTraces(proofTraces) {}

  LogicalResult matchAndRewrite(ondrix::ondsp::ReduceMacOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<ondrix::analysis::NoOverflowChunkReassociationPlan> plan =
        planConstantSaturatingReduction(op, vectorWidth, maxElements);
    if (failed(plan))
      return failure();
    auto numeric = cast<ondrix::ondsp::FixedAttr>(op.getNumeric());

    return std::move(*plan).consumeIfValid(
        op, vectorWidth,
        [&](const ondrix::ondsp::ProductSemantics &productSemantics,
            llvm::ArrayRef<llvm::APInt> validatedCoefficients, int64_t validatedWidth,
            const ondrix::analysis::NoOverflowChunkReassociationTrace &proofTrace) {
          if (productSemantics.selection != ondrix::ondsp::ProductSelection::Full ||
              validatedCoefficients.empty() || validatedWidth != vectorWidth)
            return failure();

          FailureOr<ondrix::conversion::RankOneReductionBounds> bounds =
              ondrix::conversion::createRankOneMemRefReductionBounds(
                  op, op.getLhs(), op.getRhs(), numeric.getStorage(),
                  "constant saturating memref vectorization", rewriter);
          if (failed(bounds))
            return failure();

          Location loc = op.getLoc();
          Value vectorStep = rewriter.create<arith::ConstantIndexOp>(loc, vectorWidth);
          Value remainder = rewriter.create<arith::RemUIOp>(loc, bounds->upperBound, vectorStep);
          Value vectorEnd = rewriter.create<arith::SubIOp>(loc, bounds->upperBound, remainder);
          auto vectorType = VectorType::get({vectorWidth}, numeric.getStorage());

          auto vectorLoop = rewriter.create<scf::ForOp>(
              loc, bounds->lowerBound, vectorEnd, vectorStep, ValueRange{op.getInitial()},
              [&](OpBuilder &builder, Location bodyLoc, Value base, ValueRange iterArgs) {
                Value lhs = builder.create<vector::LoadOp>(bodyLoc, vectorType, op.getLhs(), base);
                Value rhs = builder.create<vector::LoadOp>(bodyLoc, vectorType, op.getRhs(), base);
                Value next =
                    createHorizontalAccumulatorUpdate(op, iterArgs.front(), lhs, rhs, builder);
                builder.create<scf::YieldOp>(bodyLoc, next);
              });

          Value scalarStep = rewriter.create<arith::ConstantIndexOp>(loc, 1);
          auto tailLoop = rewriter.create<scf::ForOp>(
              loc, vectorEnd, bounds->upperBound, scalarStep, ValueRange{vectorLoop.getResult(0)},
              [&](OpBuilder &builder, Location bodyLoc, Value index, ValueRange iterArgs) {
                Value lhs = builder.create<memref::LoadOp>(bodyLoc, op.getLhs(), index);
                Value rhs = builder.create<memref::LoadOp>(bodyLoc, op.getRhs(), index);
                Value next = builder.create<ondrix::ondsp::MacOp>(
                    bodyLoc, iterArgs.front().getType(), iterArgs.front(), lhs, rhs, numeric,
                    *op.getProduct());
                builder.create<scf::YieldOp>(bodyLoc, next);
              });

          ondrix::analysis::NoOverflowChunkReassociationTrace recordedTrace = proofTrace;
          recordedTrace.subjectOrdinal = subjectOrdinals.lookup(op.getOperation());
          proofTraces.push_back(std::move(recordedTrace));
          rewriter.replaceOp(op, tailLoop.getResult(0));
          return success();
        });
  }

private:
  int64_t vectorWidth;
  int64_t maxElements;
  const DenseMap<Operation *, int64_t> &subjectOrdinals;
  SmallVectorImpl<ondrix::analysis::NoOverflowChunkReassociationTrace> &proofTraces;
};

LogicalResult writeProofTrace(StringRef path,
                              ArrayRef<ondrix::analysis::NoOverflowChunkReassociationTrace> traces,
                              int64_t vectorWidth, int64_t maxElements,
                              int64_t candidateReductionCount, ModuleOp module) {
  if (path.empty())
    return success();
  if (traces.empty()) {
    module.emitError("proof trace requested, but no reduction was proof-authorized");
    return failure();
  }
  if (maxElements > maxProofTraceElements) {
    module.emitError("proof trace max-elements exceeds the experimental audit limit of ")
        << maxProofTraceElements;
    return failure();
  }
  llvm::json::Array proofs;
  for (const auto &trace : traces) {
    auto hasSupportedWidth = [](const llvm::APInt &value) {
      return value.getBitWidth() <= maxProofTraceAPIntWidth;
    };
    bool supported =
        trace.coefficients.size() <= static_cast<size_t>(maxProofTraceElements) &&
        trace.originalPrefixes.size() <= static_cast<size_t>(maxProofTraceElements) + 1 &&
        trace.reassociatedPrefixes.size() <= static_cast<size_t>(maxProofTraceElements) + 1;
    for (const llvm::APInt &coefficient : trace.coefficients)
      supported &= hasSupportedWidth(coefficient);
    for (const ondrix::analysis::FixedPointRawInterval &prefix : trace.originalPrefixes)
      supported &= hasSupportedWidth(prefix.lower) && hasSupportedWidth(prefix.upper);
    for (const ondrix::analysis::FixedPointRawInterval &prefix : trace.reassociatedPrefixes)
      supported &= hasSupportedWidth(prefix.lower) && hasSupportedWidth(prefix.upper);
    if (!supported) {
      module.emitError("proof evidence exceeds the experimental audit resource limits");
      return failure();
    }
    proofs.emplace_back(ondrix::analysis::toJSON(trace));
  }
  llvm::json::Object document{{"schema_version", 1},
                              {"vector_width", vectorWidth},
                              {"analysis_max_elements", maxElements},
                              {"candidate_reduction_count", candidateReductionCount},
                              {"proofs", std::move(proofs)}};

  std::string serialized;
  llvm::raw_string_ostream serializedOutput(serialized);
  serializedOutput << llvm::json::Value(std::move(document)) << '\n';
  serializedOutput.flush();
  if (serialized.size() > maxProofTraceBytes) {
    module.emitError("proof trace exceeds the 64 MiB audit limit");
    return failure();
  }

  std::error_code error;
  llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_Text);
  if (error) {
    module.emitError("failed to open proof trace output '") << path << "': " << error.message();
    return failure();
  }
  output << serialized;
  return success();
}

class VerifyOndspConstantReassociationProofTracePass final
    : public ondrix::impl::VerifyOndspConstantReassociationProofTraceBase<
          VerifyOndspConstantReassociationProofTracePass> {
public:
  using ondrix::impl::VerifyOndspConstantReassociationProofTraceBase<
      VerifyOndspConstantReassociationProofTracePass>::
      VerifyOndspConstantReassociationProofTraceBase;

  void runOnOperation() override {
    if (proofTraceInput.empty()) {
      getOperation().emitError("proof-trace-input must not be empty");
      signalPassFailure();
      return;
    }
    if (maxElements <= 0) {
      getOperation().emitError("max-elements must be positive");
      signalPassFailure();
      return;
    }

    auto buffer = llvm::MemoryBuffer::getFile(proofTraceInput);
    if (!buffer) {
      getOperation().emitError("failed to read proof trace '")
          << proofTraceInput << "': " << buffer.getError().message();
      signalPassFailure();
      return;
    }
    if ((*buffer)->getBufferSize() > maxProofTraceBytes) {
      getOperation().emitError("proof trace exceeds the 64 MiB audit limit");
      signalPassFailure();
      return;
    }
    llvm::Expected<llvm::json::Value> parsed = llvm::json::parse((*buffer)->getBuffer());
    if (!parsed) {
      getOperation().emitError("failed to parse proof trace '")
          << proofTraceInput << "': " << llvm::toString(parsed.takeError());
      signalPassFailure();
      return;
    }
    const llvm::json::Object *document = parsed->getAsObject();
    std::optional<int64_t> schema =
        document ? document->getInteger("schema_version") : std::nullopt;
    std::optional<int64_t> vectorWidth =
        document ? document->getInteger("vector_width") : std::nullopt;
    std::optional<int64_t> analysisMaxElements =
        document ? document->getInteger("analysis_max_elements") : std::nullopt;
    std::optional<int64_t> candidateReductionCount =
        document ? document->getInteger("candidate_reduction_count") : std::nullopt;
    const llvm::json::Array *proofs = document ? document->getArray("proofs") : nullptr;
    if (!schema || *schema != 1 || !vectorWidth || *vectorWidth <= 1 || !analysisMaxElements ||
        *analysisMaxElements <= 0 || *analysisMaxElements > maxElements ||
        *analysisMaxElements > maxProofTraceElements || !candidateReductionCount ||
        *candidateReductionCount < 0 || !proofs || proofs->empty()) {
      getOperation().emitError("proof trace must contain a nonempty schema-version 1 proof array");
      signalPassFailure();
      return;
    }

    SmallVector<ondrix::ondsp::ReduceMacOp> reductions;
    getOperation().walk([&](ondrix::ondsp::ReduceMacOp op) { reductions.push_back(op); });
    if (*candidateReductionCount != static_cast<int64_t>(reductions.size()) ||
        proofs->size() > reductions.size()) {
      getOperation().emitError("proof trace candidate reduction set no longer matches the module");
      signalPassFailure();
      return;
    }

    DenseMap<int64_t, ondrix::analysis::NoOverflowChunkReassociationTrace> tracesByOrdinal;
    ondrix::analysis::NoOverflowChunkReassociationTraceParseLimits parseLimits;
    parseLimits.maxCoefficients = static_cast<size_t>(*analysisMaxElements);
    parseLimits.maxPrefixes = static_cast<size_t>(*analysisMaxElements) + 1;
    parseLimits.maxAPIntWidth = maxProofTraceAPIntWidth;
    for (const auto &[recordIndex, value] : llvm::enumerate(*proofs)) {
      FailureOr<ondrix::analysis::NoOverflowChunkReassociationTrace> trace =
          ondrix::analysis::parseNoOverflowChunkReassociationTrace(value, parseLimits);
      if (failed(trace) || trace->subjectOrdinal >= static_cast<int64_t>(reductions.size()) ||
          trace->chunkWidth != *vectorWidth ||
          !tracesByOrdinal.try_emplace(trace->subjectOrdinal, std::move(*trace)).second) {
        getOperation().emitError("invalid or duplicate proof trace record ") << recordIndex;
        signalPassFailure();
        return;
      }
    }

    for (const auto &[ordinal, reduction] : llvm::enumerate(reductions)) {
      auto trace = tracesByOrdinal.find(static_cast<int64_t>(ordinal));
      FailureOr<ondrix::analysis::NoOverflowChunkReassociationPlan> plan = failure();
      plan = planConstantSaturatingReduction(reduction, *vectorWidth, *analysisMaxElements);
      if (failed(plan)) {
        if (trace == tracesByOrdinal.end())
          continue;
        reduction.emitError("proof trace authorizes an ineligible reduction: ") << ordinal;
        signalPassFailure();
        return;
      }
      if (trace == tracesByOrdinal.end()) {
        reduction.emitError("proof trace omits a proof-authorized reduction: ") << ordinal;
        signalPassFailure();
        return;
      }

      bool matched = false;
      matched = succeeded(std::move(*plan).consumeIfValid(
          reduction, *vectorWidth,
          [&](const auto &, const auto &, int64_t,
              const ondrix::analysis::NoOverflowChunkReassociationTrace &current) {
            ondrix::analysis::NoOverflowChunkReassociationTrace rebound = current;
            rebound.subjectOrdinal = static_cast<int64_t>(ordinal);
            return succeeded(
                       ondrix::analysis::verifyNoOverflowChunkReassociationTrace(trace->second)) &&
                           ondrix::analysis::areEquivalent(trace->second, rebound)
                       ? success()
                       : failure();
          }));
      if (!matched) {
        reduction.emitError("proof trace record no longer matches this reduction: ") << ordinal;
        signalPassFailure();
        return;
      }
    }
  }
};

class ReduceMacOpVectorization final : public OpConversionPattern<ondrix::ondsp::ReduceMacOp> {
public:
  ReduceMacOpVectorization(MLIRContext *context, int64_t vectorWidth)
      : OpConversionPattern(context), vectorWidth(vectorWidth) {}

  LogicalResult matchAndRewrite(ondrix::ondsp::ReduceMacOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (hasInvalidLLVMIntegerMemorySpace(op))
      return op.emitOpError(
          "integer memory space must be nonnegative and fit in an unsigned LLVM address space");
    if (!isSupportedMemRefReduction(op))
      return failure();
    auto numeric = cast<ondrix::ondsp::FixedAttr>(op.getNumeric());
    auto elementType = cast<IntegerType>(numeric.getStorage());

    FailureOr<ondrix::conversion::RankOneReductionBounds> bounds =
        ondrix::conversion::createRankOneMemRefReductionBounds(
            op, adaptor.getLhs(), adaptor.getRhs(), elementType, "fixed-point memref vectorization",
            rewriter);
    if (failed(bounds))
      return failure();

    Location loc = op.getLoc();
    Value vectorStep = rewriter.create<arith::ConstantIndexOp>(loc, vectorWidth);
    Value remainder = rewriter.create<arith::RemUIOp>(loc, bounds->upperBound, vectorStep);
    Value vectorEnd = rewriter.create<arith::SubIOp>(loc, bounds->upperBound, remainder);
    auto vectorType = VectorType::get({vectorWidth}, elementType);

    auto vectorLoop = rewriter.create<scf::ForOp>(
        loc, bounds->lowerBound, vectorEnd, vectorStep, ValueRange{adaptor.getInitial()},
        [&](OpBuilder &builder, Location bodyLoc, Value base, ValueRange iterArgs) {
          Value lhs = builder.create<vector::LoadOp>(bodyLoc, vectorType, adaptor.getLhs(), base);
          Value rhs = builder.create<vector::LoadOp>(bodyLoc, vectorType, adaptor.getRhs(), base);
          Value next = builder.create<ondrix::ondsp::ReduceMacOp>(
              bodyLoc, iterArgs.front().getType(), iterArgs.front(), lhs, rhs, numeric,
              *op.getProduct());
          builder.create<scf::YieldOp>(bodyLoc, next);
        });

    Value scalarStep = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto tailLoop = rewriter.create<scf::ForOp>(
        loc, vectorEnd, bounds->upperBound, scalarStep, ValueRange{vectorLoop.getResult(0)},
        [&](OpBuilder &builder, Location bodyLoc, Value index, ValueRange iterArgs) {
          Value lhs = builder.create<memref::LoadOp>(bodyLoc, adaptor.getLhs(), index);
          Value rhs = builder.create<memref::LoadOp>(bodyLoc, adaptor.getRhs(), index);
          Value next = builder.create<ondrix::ondsp::MacOp>(bodyLoc, iterArgs.front().getType(),
                                                            iterArgs.front(), lhs, rhs, numeric,
                                                            *op.getProduct());
          builder.create<scf::YieldOp>(bodyLoc, next);
        });

    rewriter.replaceOp(op, tailLoop.getResult(0));
    return success();
  }

private:
  int64_t vectorWidth;
};

class VectorizeOndspFixedMemRefReducePass final
    : public ondrix::impl::VectorizeOndspFixedMemRefReduceBase<
          VectorizeOndspFixedMemRefReducePass> {
public:
  using ondrix::impl::VectorizeOndspFixedMemRefReduceBase<
      VectorizeOndspFixedMemRefReducePass>::VectorizeOndspFixedMemRefReduceBase;

  void runOnOperation() override {
    if (vectorWidth <= 0) {
      getOperation().emitError("vector-width must be positive");
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<ReduceMacOpVectorization>(&getContext(), vectorWidth);

    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, cf::ControlFlowDialect, memref::MemRefDialect,
                           ondrix::ondsp::OndspDialect, scf::SCFDialect, vector::VectorDialect>();
    target.addDynamicallyLegalOp<ondrix::ondsp::ReduceMacOp>([](ondrix::ondsp::ReduceMacOp op) {
      return !hasInvalidLLVMIntegerMemorySpace(op) && !isSupportedMemRefReduction(op);
    });

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

class VectorizeOndspConstantSaturatingMemRefReducePass final
    : public ondrix::impl::VectorizeOndspConstantSaturatingMemRefReduceBase<
          VectorizeOndspConstantSaturatingMemRefReducePass> {
public:
  using ondrix::impl::VectorizeOndspConstantSaturatingMemRefReduceBase<
      VectorizeOndspConstantSaturatingMemRefReducePass>::
      VectorizeOndspConstantSaturatingMemRefReduceBase;

  void runOnOperation() override {
    if (vectorWidth <= 1) {
      getOperation().emitError("vector-width must be greater than one");
      signalPassFailure();
      return;
    }
    if (maxElements <= 0) {
      getOperation().emitError("max-elements must be positive");
      signalPassFailure();
      return;
    }
    if (!proofTraceOutput.empty() && maxElements > maxProofTraceElements) {
      getOperation().emitError("proof trace max-elements exceeds the experimental audit limit of ")
          << maxProofTraceElements;
      signalPassFailure();
      return;
    }

    SmallVector<Operation *> reductions;
    getOperation().walk(
        [&](ondrix::ondsp::ReduceMacOp op) { reductions.push_back(op.getOperation()); });
    DenseMap<Operation *, int64_t> subjectOrdinals;
    for (const auto &[ordinal, reduction] : llvm::enumerate(reductions))
      subjectOrdinals.try_emplace(reduction, static_cast<int64_t>(ordinal));
    SmallVector<ondrix::analysis::NoOverflowChunkReassociationTrace> proofTraces;
    if (reductions.empty()) {
      if (failed(writeProofTrace(proofTraceOutput, proofTraces, vectorWidth, maxElements, 0,
                                 getOperation())))
        signalPassFailure();
      return;
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<ConstantSaturatingReduceMacVectorization>(&getContext(), vectorWidth, maxElements,
                                                           subjectOrdinals, proofTraces);

    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyOpPatternsAndFold(reductions, frozenPatterns, config)) ||
        failed(writeProofTrace(proofTraceOutput, proofTraces, vectorWidth, maxElements,
                               static_cast<int64_t>(reductions.size()), getOperation())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createVectorizeOndspFixedMemRefReducePass() {
  return std::make_unique<VectorizeOndspFixedMemRefReducePass>();
}

std::unique_ptr<Pass> ondrix::createVectorizeOndspConstantSaturatingMemRefReducePass() {
  return std::make_unique<VectorizeOndspConstantSaturatingMemRefReducePass>();
}

std::unique_ptr<Pass> ondrix::createVectorizeOndspConstantSaturatingMemRefReducePass(
    const VectorizeOndspConstantSaturatingMemRefReduceOptions &options) {
  return std::make_unique<VectorizeOndspConstantSaturatingMemRefReducePass>(options);
}

std::unique_ptr<Pass> ondrix::createVerifyOndspConstantReassociationProofTracePass() {
  return std::make_unique<VerifyOndspConstantReassociationProofTracePass>();
}

std::unique_ptr<Pass> ondrix::createVectorizeOndspFixedMemRefReducePass(
    const VectorizeOndspFixedMemRefReduceOptions &options) {
  return std::make_unique<VectorizeOndspFixedMemRefReducePass>(options);
}

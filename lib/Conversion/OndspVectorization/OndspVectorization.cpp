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
#include "mlir/Interfaces/ViewLikeInterface.h"
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
// Widest certified chunk, in machine vectors, and the widest machine vector.
// Together they bound the emitted vector type, and bounding both before they
// are multiplied is what keeps the product from overflowing.
constexpr int64_t maxChunkMultiple = 16;
constexpr int64_t maxVectorWidth = 4096;
constexpr unsigned maxProofTraceAPIntWidth = 4096;

// The emitter must refuse exactly what the replay pass refuses: a width the
// replay rejects would otherwise leave evidence this compiler produced and
// cannot revalidate.
LogicalResult checkChunkLadder(Operation *op, int64_t vectorWidth, int64_t chunkMultiple) {
  if (vectorWidth > maxVectorWidth)
    return op->emitError("vector-width exceeds the ") << maxVectorWidth << " lane limit";
  if (chunkMultiple < 1 || chunkMultiple > maxChunkMultiple)
    return op->emitError("chunk-multiple must be between 1 and ") << maxChunkMultiple;
  return success();
}

// A one-time fill dominates the reduction instead of interleaving with it, so
// only a writer under a loop the reduction also runs in competes for the same
// registers.
bool sharesEnclosingLoop(Operation *writer, Operation *reduction) {
  for (Operation *parent = reduction->getParentOp(); parent; parent = parent->getParentOp())
    if (isa<scf::ForOp, scf::WhileOp>(parent) && parent->isAncestor(writer))
      return true;
  return false;
}

// The narrowest lane count an interleaved vectorized writer already uses on the
// buffer this value views, or zero when none does. A reduction reading a buffer
// in wider lanes than the writer updating it alongside cannot keep that buffer
// in registers across both, and the repack costs more than the chunk buys.
int64_t narrowestVectorWriterLanes(Value operand, Operation *reduction) {
  Value base = operand;
  while (auto view = base.getDefiningOp<ViewLikeOpInterface>())
    base = view.getViewSource();
  int64_t narrowest = 0;
  for (Operation *user : base.getUsers()) {
    VectorType stored;
    if (auto store = dyn_cast<vector::StoreOp>(user))
      stored = store.getVectorType();
    else if (auto write = dyn_cast<vector::TransferWriteOp>(user))
      stored = write.getVectorType();
    if (!stored || stored.getRank() != 1 || stored.isScalable())
      continue;
    if (!sharesEnclosingLoop(user, reduction))
      continue;
    int64_t lanes = stored.getNumElements();
    narrowest = narrowest == 0 ? lanes : std::min(narrowest, lanes);
  }
  return narrowest;
}

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

/// How one rank-1 reduction operand is read for a vector chunk. A view whose
/// only stride is -1 is the shape a genuine convolution's reversed kernel
/// takes: `ondsp.reduce_mac` pairs its operands in increasing index order on
/// both sides, so the reversal lives in the layout rather than in the
/// operation. The span is read forward and the lanes reversed, which delivers
/// exactly the elements the ordered schedule read, in the order it read them.
struct ChunkAccess {
  Value memref;
  /// Present only for a reversed view: view index `i` is `memref[origin - i]`.
  std::optional<int64_t> origin;
  /// The static extent of a reversed view.
  int64_t size = 0;
};

std::optional<ChunkAccess> getChunkAccess(Value operand) {
  auto type = dyn_cast<MemRefType>(operand.getType());
  if (!type || type.getRank() != 1 || !hasDefaultLLVMVectorMemorySpace(type))
    return std::nullopt;
  if (isLastMemrefDimUnitStride(type))
    return ChunkAccess{operand, std::nullopt};

  // Only a subview states a reversal this pass can undo, and bufferization
  // casts that subview to the dynamic runtime contract on the way in, which
  // erases the very offset the reversal is measured from. Walk back to the
  // subview and read its own static result type.
  Value source = operand;
  while (auto cast = source.getDefiningOp<memref::CastOp>())
    source = cast.getSource();
  auto subview = source.getDefiningOp<memref::SubViewOp>();
  if (!subview)
    return std::nullopt;
  auto viewType = dyn_cast<MemRefType>(subview.getType());
  if (!viewType || viewType.getRank() != 1 || viewType.isDynamicDim(0))
    return std::nullopt;
  SmallVector<int64_t> strides;
  int64_t offset = 0;
  if (failed(getStridesAndOffset(viewType, strides, offset)) || strides.size() != 1 ||
      strides[0] != -1 || ShapedType::isDynamic(offset))
    return std::nullopt;
  auto sourceType = dyn_cast<MemRefType>(subview.getSource().getType());
  if (!sourceType || sourceType.getRank() != 1 || sourceType.isDynamicDim(0) ||
      !isLastMemrefDimUnitStride(sourceType) || !hasDefaultLLVMVectorMemorySpace(sourceType))
    return std::nullopt;
  // The reversed span must lie inside the source: view index i reads
  // `source[offset - i]`, so the last index read is `offset - (size - 1)`.
  if (offset < 0 || offset >= sourceType.getDimSize(0) || viewType.getDimSize(0) > offset + 1)
    return std::nullopt;
  return ChunkAccess{subview.getSource(), offset, viewType.getDimSize(0)};
}

/// Loads one chunk of the operand starting at view index `base`.
Value loadReductionChunk(const ChunkAccess &access, Value base, VectorType vectorType, Location loc,
                         OpBuilder &builder) {
  if (!access.origin)
    return builder.create<vector::LoadOp>(loc, vectorType, access.memref, base);
  // View index i is source index origin - i, so a chunk covering view
  // [base, base + W) is the source span ending at origin - base. The load runs
  // forward from that span's start and lane i then takes span element W-1-i.
  int64_t lanes = vectorType.getNumElements();
  Value spanEnd = builder.create<arith::ConstantIndexOp>(loc, *access.origin - (lanes - 1));
  Value spanBase = builder.create<arith::SubIOp>(loc, spanEnd, base);
  Value span = builder.create<vector::LoadOp>(loc, vectorType, access.memref, spanBase);
  SmallVector<int64_t> reversed;
  for (int64_t lane = 0; lane < lanes; ++lane)
    reversed.push_back(lanes - 1 - lane);
  return builder.create<vector::ShuffleOp>(loc, span, span, reversed);
}

bool hasSupportedReductionShape(ondrix::ondsp::ReduceMacOp op, bool allowReversedLayout) {
  auto accumulator = dyn_cast<ondrix::ondsp::AccType>(op.getInitial().getType());
  auto numeric = dyn_cast<ondrix::ondsp::FixedAttr>(op.getNumeric());
  auto lhsType = dyn_cast<MemRefType>(op.getLhs().getType());
  auto rhsType = dyn_cast<MemRefType>(op.getRhs().getType());
  bool layoutOk = allowReversedLayout ? getChunkAccess(op.getLhs()).has_value() &&
                                            getChunkAccess(op.getRhs()).has_value()
                                      : lhsType && rhsType && isLastMemrefDimUnitStride(lhsType) &&
                                            isLastMemrefDimUnitStride(rhsType);
  return accumulator && numeric && op.getProduct() && lhsType && rhsType &&
         lhsType.getRank() == 1 && rhsType.getRank() == 1 &&
         lhsType.getElementType() == numeric.getStorage() &&
         rhsType.getElementType() == numeric.getStorage() &&
         hasDefaultLLVMVectorMemorySpace(lhsType) && hasDefaultLLVMVectorMemorySpace(rhsType) &&
         layoutOk &&
         ondrix::conversion::isSupportedFixedVectorMacDomain(accumulator, numeric,
                                                             *op.getProduct());
}

/// Both routes admit a reversed operand: the reversal is undone in the load,
/// and the constant certificate is taken over the coefficients in read order.
bool isVectorizableMemRefReduction(ondrix::ondsp::ReduceMacOp op) {
  return hasSupportedReductionShape(op, /*allowReversedLayout=*/true);
}

/// One chunk's products folded into the accumulator. Adjacent products are
/// pre-added in i32 when the trace certifies the pair width, the shape the
/// backends select a packed multiply-add for; the chunk sum is then formed in
/// i32 when the trace certifies that width too, and widened to i64 otherwise.
Value createHorizontalAccumulatorUpdate(
    ondrix::ondsp::ReduceMacOp op, Value accumulator, Value lhs, Value rhs,
    const ondrix::analysis::NoOverflowChunkReassociationTrace &trace, OpBuilder &builder) {
  auto accumulatorType = cast<ondrix::ondsp::AccType>(accumulator.getType());
  auto numeric = cast<ondrix::ondsp::FixedAttr>(op.getNumeric());
  FailureOr<ondrix::conversion::FixedVectorProductTerms> terms =
      ondrix::conversion::lowerFixedVectorProductTerms(op, accumulatorType, numeric,
                                                       *op.getProduct(), lhs, rhs, builder);
  assert(succeeded(terms) && "validated fixed Vector product domain must lower");
  auto termType = cast<VectorType>(terms->getTerms().getType());
  Location loc = op.getLoc();
  if (trace.pairTermWidth == 32 && termType.getElementTypeBitWidth() == 32 &&
      termType.getNumElements() % 2 == 0) {
    SmallVector<int64_t> even, odd;
    for (int64_t lane = 0; lane < termType.getNumElements(); lane += 2) {
      even.push_back(lane);
      odd.push_back(lane + 1);
    }
    Value evens =
        builder.create<vector::ShuffleOp>(loc, terms->getTerms(), terms->getTerms(), even);
    Value odds = builder.create<vector::ShuffleOp>(loc, terms->getTerms(), terms->getTerms(), odd);
    Value pairs = builder.create<arith::AddIOp>(loc, evens, odds);
    Type sumStorage = builder.getI32Type();
    if (trace.implementationTermWidth != 32) {
      sumStorage = builder.getI64Type();
      pairs = builder.create<arith::ExtSIOp>(
          loc, VectorType::get({termType.getNumElements() / 2}, sumStorage), pairs);
    }
    Value sum = builder.create<vector::ReductionOp>(loc, vector::CombiningKind::ADD, pairs);
    auto sumNumeric =
        ondrix::ondsp::FixedAttr::get(builder.getContext(), terms->getNumeric().getSignedness(),
                                      sumStorage, terms->getNumeric().getFrac());
    return builder.create<ondrix::ondsp::AccAddTermOp>(loc, accumulator.getType(), accumulator, sum,
                                                       sumNumeric);
  }
  FailureOr<ondrix::conversion::FixedVectorHorizontalSum> horizontal =
      ondrix::conversion::lowerFixedVectorHorizontalSum(op, *terms, builder);
  assert(succeeded(horizontal) && "validated fixed Vector horizontal sum must lower");
  return builder.create<ondrix::ondsp::AccAddTermOp>(
      op.getLoc(), accumulator.getType(), accumulator, horizontal->sum, horizontal->numeric);
}

FailureOr<ondrix::analysis::NoOverflowChunkReassociationPlan>
planConstantSaturatingReduction(ondrix::ondsp::ReduceMacOp op, int64_t vectorWidth,
                                int64_t maxElements) {
  if (!isVectorizableMemRefReduction(op) ||
      !op.getInitial().getDefiningOp<ondrix::ondsp::AccZeroOp>())
    return failure();

  auto accumulator = cast<ondrix::ondsp::AccType>(op.getInitial().getType());
  auto numeric = cast<ondrix::ondsp::FixedAttr>(op.getNumeric());
  if (!ondrix::ondsp::isFullProduct(*op.getProduct()) ||
      !ondrix::conversion::isSupportedFixedHorizontalMacDomain(accumulator, numeric,
                                                               *op.getProduct()))
    return failure();
  FailureOr<ondrix::conversion::SupportedFixedMacDomain> domain =
      ondrix::conversion::getSupportedFixedVectorMacDomain(op, accumulator, numeric,
                                                           *op.getProduct());
  if (failed(domain) || domain->termStorage.getWidth() > 64)
    return failure();

  std::optional<ChunkAccess> access = getChunkAccess(op.getRhs());
  if (!access)
    return failure();
  FailureOr<ondrix::ConstantIntegerMemRefFacts> constant =
      ondrix::analyzeConstantIntegerMemRef(access->memref, maxElements);
  if (failed(constant))
    return failure();
  if (!access->origin) {
    auto rhsType = cast<MemRefType>(op.getRhs().getType());
    if (!rhsType.isDynamicDim(0) &&
        constant->getSequence().getElementCount() != rhsType.getDimSize(0))
      return failure();
    return ondrix::analysis::FixedPointPrefixRangePlanner::planZeroSeededConstantChunkReduction(
        op, *constant, vectorWidth);
  }
  // A reversed view reads `source[origin - i]`; the certificate is over the
  // coefficients in that reading order, and the plan is bound to the view.
  ArrayRef<llvm::APInt> values = constant->getSequence().getValues();
  if (*access->origin >= static_cast<int64_t>(values.size()) || access->size > *access->origin + 1)
    return failure();
  SmallVector<llvm::APInt> readOrder;
  for (int64_t index = 0; index < access->size; ++index)
    readOrder.push_back(values[*access->origin - index]);
  return ondrix::analysis::FixedPointPrefixRangePlanner::planZeroSeededConstantChunkReduction(
      op, readOrder, op.getRhs(), vectorWidth);
}

/// The certified chunk may be several machine vectors wide. Widening it does
/// not weaken the obligation - the analysis re-derives every prefix bound for
/// the width it is asked about - but it does move work out of the loop-carried
/// recurrence, so the largest authorized width in the ladder is taken and a
/// subject too short for it falls back rather than losing vectorization.
struct SelectedConstantSaturatingPlan {
  ondrix::analysis::NoOverflowChunkReassociationPlan plan;
  int64_t chunkWidth;
};

FailureOr<SelectedConstantSaturatingPlan>
selectConstantSaturatingPlan(ondrix::ondsp::ReduceMacOp op, int64_t vectorWidth,
                             int64_t chunkMultiple, int64_t maxElements) {
  for (int64_t multiple = chunkMultiple; multiple >= 1; --multiple) {
    int64_t chunkWidth = vectorWidth * multiple;
    FailureOr<ondrix::analysis::NoOverflowChunkReassociationPlan> plan =
        planConstantSaturatingReduction(op, chunkWidth, maxElements);
    if (succeeded(plan))
      return SelectedConstantSaturatingPlan{std::move(*plan), chunkWidth};
  }
  return failure();
}

class ConstantSaturatingReduceMacVectorization final
    : public OpRewritePattern<ondrix::ondsp::ReduceMacOp> {
public:
  ConstantSaturatingReduceMacVectorization(
      MLIRContext *context, int64_t vectorWidth, int64_t chunkMultiple, int64_t maxElements,
      bool dischargeUpdateGuard, const DenseMap<Operation *, int64_t> &subjectOrdinals,
      SmallVectorImpl<ondrix::analysis::NoOverflowChunkReassociationTrace> &proofTraces)
      : OpRewritePattern(context), vectorWidth(vectorWidth), chunkMultiple(chunkMultiple),
        maxElements(maxElements), dischargeUpdateGuard(dischargeUpdateGuard),
        subjectOrdinals(subjectOrdinals), proofTraces(proofTraces) {}

  LogicalResult matchAndRewrite(ondrix::ondsp::ReduceMacOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<SelectedConstantSaturatingPlan> selected =
        selectConstantSaturatingPlan(op, vectorWidth, chunkMultiple, maxElements);
    if (failed(selected))
      return failure();
    int64_t chunkWidth = selected->chunkWidth;
    auto numeric = cast<ondrix::ondsp::FixedAttr>(op.getNumeric());

    return std::move(selected->plan)
        .consumeIfValid(
            op, chunkWidth,
            [&](const ondrix::ondsp::ProductSemantics &productSemantics,
                llvm::ArrayRef<llvm::APInt> validatedCoefficients, int64_t validatedWidth,
                const ondrix::analysis::NoOverflowChunkReassociationTrace &proofTrace) {
              if (productSemantics.selection != ondrix::ondsp::ProductSelection::Full ||
                  validatedCoefficients.empty() || validatedWidth != chunkWidth)
                return failure();

              FailureOr<ondrix::conversion::RankOneReductionBounds> bounds =
                  ondrix::conversion::createRankOneMemRefReductionBounds(
                      op, op.getLhs(), op.getRhs(), numeric.getStorage(),
                      "constant saturating memref vectorization", rewriter);
              if (failed(bounds))
                return failure();

              Location loc = op.getLoc();
              Value seed = createCertifiedSeed(op, rewriter);
              Value vectorStep = rewriter.create<arith::ConstantIndexOp>(loc, chunkWidth);
              Value remainder =
                  rewriter.create<arith::RemUIOp>(loc, bounds->upperBound, vectorStep);
              Value vectorEnd = rewriter.create<arith::SubIOp>(loc, bounds->upperBound, remainder);
              auto vectorType = VectorType::get({chunkWidth}, numeric.getStorage());

              auto vectorLoop = rewriter.create<scf::ForOp>(
                  loc, bounds->lowerBound, vectorEnd, vectorStep, ValueRange{seed},
                  [&](OpBuilder &builder, Location bodyLoc, Value base, ValueRange iterArgs) {
                    Value lhs = loadReductionChunk(*getChunkAccess(op.getLhs()), base, vectorType,
                                                   bodyLoc, builder);
                    Value rhs = loadReductionChunk(*getChunkAccess(op.getRhs()), base, vectorType,
                                                   bodyLoc, builder);
                    Value next = createHorizontalAccumulatorUpdate(op, iterArgs.front(), lhs, rhs,
                                                                   proofTrace, builder);
                    builder.create<scf::YieldOp>(bodyLoc, next);
                  });

              Value scalarStep = rewriter.create<arith::ConstantIndexOp>(loc, 1);
              auto tailLoop = rewriter.create<scf::ForOp>(
                  loc, vectorEnd, bounds->upperBound, scalarStep,
                  ValueRange{vectorLoop.getResult(0)},
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
  // The plan proves that no prefix of the emitted schedule reaches the
  // accumulator rail, so its declared saturating update is unreachable. Seed
  // the certified schedule with the wrapping accumulator of the same storage,
  // frac and signedness whenever every consumer only reads the final value.
  Value createCertifiedSeed(ondrix::ondsp::ReduceMacOp op, PatternRewriter &rewriter) const {
    auto declared = cast<ondrix::ondsp::AccType>(op.getInitial().getType());
    if (!dischargeUpdateGuard || !llvm::all_of(op.getResult().getUsers(), [](Operation *user) {
          return isa<ondrix::ondsp::AccExportOp>(user);
        }))
      return op.getInitial();
    auto certified = ondrix::ondsp::AccType::get(
        declared.getContext(), declared.getStorage(), declared.getFrac(), declared.getSignedness(),
        ondrix::ondsp::OverflowMode::Wrap, declared.getLanes());
    return rewriter.create<ondrix::ondsp::AccZeroOp>(op.getLoc(), certified);
  }

  int64_t vectorWidth;
  int64_t chunkMultiple;
  int64_t maxElements;
  bool dischargeUpdateGuard;
  const DenseMap<Operation *, int64_t> &subjectOrdinals;
  SmallVectorImpl<ondrix::analysis::NoOverflowChunkReassociationTrace> &proofTraces;
};

LogicalResult writeProofTrace(StringRef path,
                              ArrayRef<ondrix::analysis::NoOverflowChunkReassociationTrace> traces,
                              int64_t vectorWidth, int64_t chunkMultiple, int64_t maxElements,
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
  llvm::json::Object document{{"schema_version", 2},
                              {"vector_width", vectorWidth},
                              {"chunk_multiple", chunkMultiple},
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
    std::optional<int64_t> traceChunkMultiple =
        document ? document->getInteger("chunk_multiple") : std::nullopt;
    std::optional<int64_t> analysisMaxElements =
        document ? document->getInteger("analysis_max_elements") : std::nullopt;
    std::optional<int64_t> candidateReductionCount =
        document ? document->getInteger("candidate_reduction_count") : std::nullopt;
    const llvm::json::Array *proofs = document ? document->getArray("proofs") : nullptr;
    if (!schema || *schema != 2 || !vectorWidth || *vectorWidth <= 1 || !traceChunkMultiple ||
        *traceChunkMultiple < 1 || *traceChunkMultiple > maxChunkMultiple || !analysisMaxElements ||
        *analysisMaxElements <= 0 || *analysisMaxElements > maxElements ||
        *analysisMaxElements > maxProofTraceElements || !candidateReductionCount ||
        *candidateReductionCount < 0 || !proofs || proofs->empty()) {
      getOperation().emitError("proof trace must contain a nonempty schema-version 2 proof array");
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
          trace->chunkWidth % *vectorWidth != 0 ||
          trace->chunkWidth / *vectorWidth > *traceChunkMultiple ||
          !tracesByOrdinal.try_emplace(trace->subjectOrdinal, std::move(*trace)).second) {
        getOperation().emitError("invalid or duplicate proof trace record ") << recordIndex;
        signalPassFailure();
        return;
      }
    }

    for (const auto &[ordinal, reduction] : llvm::enumerate(reductions)) {
      auto trace = tracesByOrdinal.find(static_cast<int64_t>(ordinal));
      // Rerun the emitting pass's own width ladder: a record naming a narrower
      // chunk than the ladder would select is a divergence, not a weaker claim.
      FailureOr<SelectedConstantSaturatingPlan> selected = selectConstantSaturatingPlan(
          reduction, *vectorWidth, *traceChunkMultiple, *analysisMaxElements);
      FailureOr<ondrix::analysis::NoOverflowChunkReassociationPlan> plan = failure();
      int64_t chunkWidth = *vectorWidth;
      if (succeeded(selected)) {
        chunkWidth = selected->chunkWidth;
        plan = std::move(selected->plan);
      }
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
      if (trace->second.chunkWidth != chunkWidth) {
        reduction.emitError("proof trace record names a chunk width the selection would not "
                            "choose: ")
            << ordinal;
        signalPassFailure();
        return;
      }

      bool matched = false;
      matched = succeeded(std::move(*plan).consumeIfValid(
          reduction, chunkWidth,
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
  ReduceMacOpVectorization(MLIRContext *context, int64_t vectorWidth, int64_t chunkMultiple,
                           bool pairFoldSquares, bool requantizedProducts)
      : OpConversionPattern(context), vectorWidth(vectorWidth), chunkMultiple(chunkMultiple),
        pairFoldSquares(pairFoldSquares), requantizedProducts(requantizedProducts) {}

  LogicalResult matchAndRewrite(ondrix::ondsp::ReduceMacOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (hasInvalidLLVMIntegerMemorySpace(op))
      return op.emitOpError(
          "integer memory space must be nonnegative and fit in an unsigned LLVM address space");
    if (!isVectorizableMemRefReduction(op))
      return failure();
    if (!requantizedProducts && op.getProduct()->getShift() != 0)
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
    // A wider chunk only helps while the reduction can fill one: past that the
    // vector loop is empty and the whole reduction lands in the scalar tail,
    // so the width steps down to what the static extent actually admits.
    int64_t chunkWidth = vectorWidth * chunkMultiple;
    auto lhsType = cast<MemRefType>(op.getLhs().getType());
    int64_t extent = lhsType.isDynamicDim(0) ? 0 : lhsType.getDimSize(0);
    while (chunkWidth > vectorWidth && (extent == 0 || extent < chunkWidth))
      chunkWidth -= vectorWidth;
    // An adapting state another stage already updates in narrower lanes is the
    // other bound: agreeing with that writer keeps the buffer in registers.
    for (Value operand : {op.getLhs(), op.getRhs()}) {
      int64_t writerWidth = narrowestVectorWriterLanes(operand, op);
      while (writerWidth > 0 && chunkWidth > vectorWidth && chunkWidth > writerWidth)
        chunkWidth -= vectorWidth;
    }
    Value vectorStep = rewriter.create<arith::ConstantIndexOp>(loc, chunkWidth);
    Value remainder = rewriter.create<arith::RemUIOp>(loc, bounds->upperBound, vectorStep);
    Value vectorEnd = rewriter.create<arith::SubIOp>(loc, bounds->upperBound, remainder);
    auto vectorType = VectorType::get({chunkWidth}, elementType);
    auto loadChunks = [&](OpBuilder &builder, Location bodyLoc, Value base) {
      Value lhs =
          loadReductionChunk(*getChunkAccess(adaptor.getLhs()), base, vectorType, bodyLoc, builder);
      Value rhs =
          loadReductionChunk(*getChunkAccess(adaptor.getRhs()), base, vectorType, bodyLoc, builder);
      return std::pair<Value, Value>{lhs, rhs};
    };

    auto accumulator = cast<ondrix::ondsp::AccType>(op.getInitial().getType());
    bool exactModulo =
        ondrix::ondsp::classifyReductionReassociation(accumulator.getUpdateOverflow()) ==
            ondrix::ondsp::ReductionReassociationSafety::ExactModulo &&
        ondrix::conversion::isSupportedFixedHorizontalMacDomain(accumulator, numeric,
                                                                *op.getProduct());
    // A reduction of a sequence against itself with 32-bit full products (a
    // Q15 sum of squares) folds pairs of terms in i32 before widening, where
    // the declared target folds adjacent products in its multiply-add.
    bool squarePairs = pairFoldSquares && exactModulo && adaptor.getLhs() == adaptor.getRhs() &&
                       op.getProduct()->getSelection() == ondrix::ondsp::ProductSelection::Full &&
                       elementType.getWidth() <= 16 && chunkWidth % (2 * vectorWidth) == 0;
    Value vectorResult;
    if (exactModulo) {
      // Wrapping updates commute modulo 2^W, so one machine vector of i64 lane
      // sums carried to the loop exit and folded once equals the ordered fold.
      auto laneType = VectorType::get({vectorWidth}, rewriter.getI64Type());
      Value zeroLanes =
          rewriter.create<arith::ConstantOp>(loc, laneType, rewriter.getZeroAttr(laneType));
      auto laneLoop = rewriter.create<scf::ForOp>(
          loc, bounds->lowerBound, vectorEnd, vectorStep, ValueRange{zeroLanes},
          [&](OpBuilder &builder, Location bodyLoc, Value base, ValueRange iterArgs) {
            auto [lhs, rhs] = loadChunks(builder, bodyLoc, base);
            Value lanes = iterArgs.front();
            if (squarePairs) {
              // Squares are nonnegative, so two of them sum below 2^31 + 1: the
              // i32 pair sum read unsigned is exact, including the one carry.
              // The products are formed per pair slice so the backend sees one
              // multiply feeding one even/odd fold.
              SmallVector<int64_t> even, odd;
              for (int64_t lane = 0; lane < vectorWidth; ++lane) {
                even.push_back(2 * lane);
                odd.push_back(2 * lane + 1);
              }
              for (int64_t offset = 0; offset < chunkWidth; offset += 2 * vectorWidth) {
                Value slice = lhs;
                if (chunkWidth != 2 * vectorWidth)
                  slice = builder.create<vector::ExtractStridedSliceOp>(
                      bodyLoc, slice, ArrayRef<int64_t>{offset}, ArrayRef<int64_t>{2 * vectorWidth},
                      ArrayRef<int64_t>{1});
                FailureOr<ondrix::conversion::FixedVectorProductTerms> sliceTerms =
                    ondrix::conversion::lowerFixedVectorProductTerms(
                        op, accumulator, numeric, *op.getProduct(), slice, slice, builder);
                assert(succeeded(sliceTerms) && "validated fixed Vector product domain must lower");
                Value products = sliceTerms->getTerms();
                Value evens = builder.create<vector::ShuffleOp>(bodyLoc, products, products, even);
                Value odds = builder.create<vector::ShuffleOp>(bodyLoc, products, products, odd);
                Value pairs = builder.create<arith::AddIOp>(bodyLoc, evens, odds);
                Value widened = builder.create<arith::ExtUIOp>(bodyLoc, laneType, pairs);
                lanes = builder.create<arith::AddIOp>(bodyLoc, lanes, widened);
              }
              builder.create<scf::YieldOp>(bodyLoc, lanes);
              return;
            }
            FailureOr<ondrix::conversion::FixedVectorProductTerms> terms =
                ondrix::conversion::lowerFixedVectorProductTerms(
                    op, accumulator, numeric, *op.getProduct(), lhs, rhs, builder);
            assert(succeeded(terms) && "validated fixed Vector product domain must lower");
            for (int64_t offset = 0; offset < chunkWidth; offset += vectorWidth) {
              Value slice = terms->getTerms();
              if (chunkWidth != vectorWidth)
                slice = builder.create<vector::ExtractStridedSliceOp>(
                    bodyLoc, slice, ArrayRef<int64_t>{offset}, ArrayRef<int64_t>{vectorWidth},
                    ArrayRef<int64_t>{1});
              if (slice.getType() != laneType)
                slice = builder.create<arith::ExtSIOp>(bodyLoc, laneType, slice);
              lanes = builder.create<arith::AddIOp>(bodyLoc, lanes, slice);
            }
            builder.create<scf::YieldOp>(bodyLoc, lanes);
          });
      Value sum = rewriter.create<vector::ReductionOp>(loc, vector::CombiningKind::ADD,
                                                       laneLoop.getResult(0));
      auto termNumeric =
          ondrix::ondsp::FixedAttr::get(rewriter.getContext(), ondrix::ondsp::Signedness::Signed,
                                        rewriter.getI64Type(), accumulator.getFrac());
      vectorResult = rewriter.create<ondrix::ondsp::AccAddTermOp>(
          loc, accumulator, adaptor.getInitial(), sum, termNumeric);
    } else {
      auto vectorLoop = rewriter.create<scf::ForOp>(
          loc, bounds->lowerBound, vectorEnd, vectorStep, ValueRange{adaptor.getInitial()},
          [&](OpBuilder &builder, Location bodyLoc, Value base, ValueRange iterArgs) {
            auto [lhs, rhs] = loadChunks(builder, bodyLoc, base);
            Value next = builder.create<ondrix::ondsp::ReduceMacOp>(
                bodyLoc, iterArgs.front().getType(), iterArgs.front(), lhs, rhs, numeric,
                *op.getProduct());
            builder.create<scf::YieldOp>(bodyLoc, next);
          });
      vectorResult = vectorLoop.getResult(0);
    }

    Value scalarStep = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto tailLoop = rewriter.create<scf::ForOp>(
        loc, vectorEnd, bounds->upperBound, scalarStep, ValueRange{vectorResult},
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
  int64_t chunkMultiple;
  bool pairFoldSquares;
  bool requantizedProducts;
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
    if (failed(checkChunkLadder(getOperation(), vectorWidth, chunkMultiple))) {
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<ReduceMacOpVectorization>(&getContext(), vectorWidth, chunkMultiple,
                                           pairFoldSquares, requantizedProducts);

    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, cf::ControlFlowDialect, memref::MemRefDialect,
                           ondrix::ondsp::OndspDialect, scf::SCFDialect, vector::VectorDialect>();
    bool keepRequantized = !requantizedProducts;
    target.addDynamicallyLegalOp<ondrix::ondsp::ReduceMacOp>(
        [keepRequantized](ondrix::ondsp::ReduceMacOp op) {
          return !hasInvalidLLVMIntegerMemorySpace(op) &&
                 (!isVectorizableMemRefReduction(op) ||
                  (keepRequantized && op.getProduct()->getShift() != 0));
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
    if (failed(checkChunkLadder(getOperation(), vectorWidth, chunkMultiple))) {
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
      if (failed(writeProofTrace(proofTraceOutput, proofTraces, vectorWidth, chunkMultiple,
                                 maxElements, 0, getOperation())))
        signalPassFailure();
      return;
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<ConstantSaturatingReduceMacVectorization>(
        &getContext(), vectorWidth, chunkMultiple, maxElements, dischargeUpdateGuard,
        subjectOrdinals, proofTraces);

    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyOpPatternsAndFold(reductions, frozenPatterns, config)) ||
        failed(writeProofTrace(proofTraceOutput, proofTraces, vectorWidth, chunkMultiple,
                               maxElements, static_cast<int64_t>(reductions.size()),
                               getOperation())))
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

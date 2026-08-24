#include "ondrix/Conversion/OndspVectorization/OndspVectorization.h"
#include "ondrix/Conversion/Utils/MemRefLayoutUtils.h"
#include "ondrix/Conversion/Utils/ReductionUtils.h"

#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace ondrix {
#define GEN_PASS_DEF_VECTORIZEONDSPFPFASTMEMREFREDUCE
#include "ondrix/Conversion/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

/// Largest accepted lane count, bounding every index the rewrite derives.
constexpr int64_t kMaxVectorWidth = 4096;

/// Largest accepted chain count; beyond this the iter-arg pressure exceeds
/// any real register file.
constexpr int64_t kMaxInterleave = 64;

/// Reduction length when both operand extents are known at compile time.
std::optional<int64_t> getStaticReductionLength(MemRefType lhsType, MemRefType rhsType) {
  if (!lhsType.isDynamicDim(0))
    return lhsType.getDimSize(0);
  if (!rhsType.isDynamicDim(0))
    return rhsType.getDimSize(0);
  return std::nullopt;
}

bool isSupportedFastMemRefReduction(ondrix::ondsp::ReduceMacOp op, int64_t vectorWidth) {
  auto numeric = dyn_cast<ondrix::ondsp::FpAttr>(op.getNumeric());
  if (!numeric || !numeric.getFormat().isF32() ||
      numeric.getContract() != ondrix::ondsp::FpContractMode::Fast)
    return false;
  // A verified f32 reduction already has the rank, element type and absent
  // product; only the layout facts the Vector lowering needs are this pass's
  // obligation. The initial value is carried, not reproduced, so it is free.
  auto lhsType = dyn_cast<MemRefType>(op.getLhs().getType());
  auto rhsType = dyn_cast<MemRefType>(op.getRhs().getType());
  if (!lhsType || !rhsType || !ondrix::conversion::hasDefaultLLVMVectorMemorySpace(lhsType) ||
      !ondrix::conversion::hasDefaultLLVMVectorMemorySpace(rhsType) ||
      !isLastMemrefDimUnitStride(lhsType) || !isLastMemrefDimUnitStride(rhsType))
    return false;
  // A statically short reduction has no lane to fill, so it keeps the ordered
  // schedule outright instead of carrying a branch that can never be taken.
  std::optional<int64_t> length = getStaticReductionLength(lhsType, rhsType);
  return !length || *length >= vectorWidth;
}

class FastReduceMacOpVectorization final : public OpConversionPattern<ondrix::ondsp::ReduceMacOp> {
public:
  FastReduceMacOpVectorization(MLIRContext *context, int64_t vectorWidth, bool fuseTerms,
                               int64_t interleave)
      : OpConversionPattern(context), vectorWidth(vectorWidth), fuseTerms(fuseTerms),
        interleave(interleave) {}

  LogicalResult matchAndRewrite(ondrix::ondsp::ReduceMacOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!isSupportedFastMemRefReduction(op, vectorWidth))
      return failure();

    Type elementType = rewriter.getF32Type();
    FailureOr<ondrix::conversion::RankOneReductionBounds> bounds =
        ondrix::conversion::createRankOneMemRefReductionBounds(
            op, adaptor.getLhs(), adaptor.getRhs(), elementType, "fast f32 memref vectorization",
            rewriter);
    if (failed(bounds))
      return failure();

    Location loc = op.getLoc();
    Value vectorStep = rewriter.create<arith::ConstantIndexOp>(loc, vectorWidth);
    Value scalarStep = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value remainder = rewriter.create<arith::RemUIOp>(loc, bounds->upperBound, vectorStep);
    Value vectorEnd = rewriter.create<arith::SubIOp>(loc, bounds->upperBound, remainder);
    auto vectorType = VectorType::get({vectorWidth}, elementType);

    auto blockBase = [&](OpBuilder &builder, Location branchLoc, Value from, int64_t blockIndex) {
      if (blockIndex == 0)
        return from;
      Value offset = builder.create<arith::ConstantIndexOp>(branchLoc, blockIndex * vectorWidth);
      return builder.create<arith::AddIOp>(branchLoc, from, offset).getResult();
    };

    auto loadProducts = [&](OpBuilder &builder, Location branchLoc, Value base) {
      Value lhs = builder.create<vector::LoadOp>(branchLoc, vectorType, adaptor.getLhs(), base);
      Value rhs = builder.create<vector::LoadOp>(branchLoc, vectorType, adaptor.getRhs(), base);
      return std::pair<Value, Value>(lhs, rhs);
    };

    auto buildBatched = [&](OpBuilder &builder, Location branchLoc, int64_t chains) {
      // Every chain's lane seed is a real block of W products of the source
      // reduction, never a synthesized identity.
      SmallVector<Value> partials;
      for (int64_t chain = 0; chain < chains; ++chain) {
        auto [lhs, rhs] = loadProducts(builder, branchLoc,
                                       blockBase(builder, branchLoc, bounds->lowerBound, chain));
        partials.push_back(builder.create<arith::MulFOp>(branchLoc, lhs, rhs));
      }

      Value groupStep = builder.create<arith::ConstantIndexOp>(branchLoc, chains * vectorWidth);
      Value firstGroup = blockBase(builder, branchLoc, bounds->lowerBound, chains);
      // With chains > 1 the extent is static, so the group range is exact and
      // the leftover blocks below are compile-time counted.
      Value groupEnd = vectorEnd;
      int64_t leftoverBlocks = 0;
      if (chains > 1) {
        int64_t length = *getStaticReductionLength(bounds->lhsType, bounds->rhsType);
        int64_t blocks = length / vectorWidth;
        leftoverBlocks = (blocks - chains) % chains;
        groupEnd = blockBase(builder, branchLoc, bounds->lowerBound, blocks - leftoverBlocks);
      }
      auto vectorLoop = builder.create<scf::ForOp>(
          branchLoc, firstGroup, groupEnd, groupStep, partials,
          [&](OpBuilder &bodyBuilder, Location bodyLoc, Value base, ValueRange iterArgs) {
            SmallVector<Value> next;
            for (int64_t chain = 0; chain < chains; ++chain) {
              auto [lhs, rhs] =
                  loadProducts(bodyBuilder, bodyLoc, blockBase(bodyBuilder, bodyLoc, base, chain));
              next.push_back(accumulateTerm(bodyLoc, lhs, rhs, iterArgs[chain], bodyBuilder));
            }
            bodyBuilder.create<scf::YieldOp>(bodyLoc, next);
          });
      partials.assign(vectorLoop.getResults().begin(), vectorLoop.getResults().end());

      for (int64_t block = 0; block < leftoverBlocks; ++block) {
        auto [lhs, rhs] =
            loadProducts(builder, branchLoc, blockBase(builder, branchLoc, groupEnd, block));
        partials[block] = accumulateTerm(branchLoc, lhs, rhs, partials[block], builder);
      }

      // The chain partials are interior nodes of the one rebuilt tree, merged
      // pairwise; the permission is recorded once, at the fold below.
      while (partials.size() > 1) {
        SmallVector<Value> merged;
        for (size_t i = 0; i + 1 < partials.size(); i += 2)
          merged.push_back(builder.create<arith::AddFOp>(branchLoc, partials[i], partials[i + 1]));
        if (partials.size() % 2 != 0)
          merged.push_back(partials.back());
        partials = std::move(merged);
      }

      // The initial is the fold's accumulator, so the seed enters exactly once
      // and stays the first leaf: an implicit +0.0 is no leaf of the source
      // tree, and it turns an all-negative-zero reduction's declared -0.0 into
      // +0.0. The fold is also where R is recorded, since it exists only
      // because the tree was rebuilt.
      Value folded = ondrix::ondsp::consumeFastPermission(
          builder.create<vector::ReductionOp>(branchLoc, vector::CombiningKind::ADD,
                                              partials.front(), adaptor.getInitial()),
          ondrix::ondsp::FastPermission::RebuildReductionTree);
      return createOrderedTail(branchLoc, adaptor, vectorEnd, bounds->upperBound, scalarStep,
                               folded, builder);
    };

    // Padding up to one block would be the term invention this rewrite exists
    // to avoid. Only a dynamic extent needs the branch: a statically short one
    // never reaches this pattern. Interleaving needs the compile-time block
    // count, so a dynamic extent keeps the single chain.
    if (std::optional<int64_t> length =
            getStaticReductionLength(bounds->lhsType, bounds->rhsType)) {
      int64_t chains = std::min(interleave, *length / vectorWidth);
      rewriter.replaceOp(op, buildBatched(rewriter, loc, chains));
      return success();
    }

    Value hasBlock = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, vectorEnd,
                                                    bounds->lowerBound);
    auto guarded = rewriter.create<scf::IfOp>(
        loc, hasBlock,
        [&](OpBuilder &builder, Location branchLoc) {
          builder.create<scf::YieldOp>(branchLoc, buildBatched(builder, branchLoc, 1));
        },
        [&](OpBuilder &builder, Location branchLoc) {
          builder.create<scf::YieldOp>(branchLoc,
                                       createOrderedTail(branchLoc, adaptor, bounds->lowerBound,
                                                         bounds->upperBound, scalarStep,
                                                         adaptor.getInitial(), builder));
        });

    rewriter.replaceOp(op, guarded.getResult(0));
    return success();
  }

private:
  /// Folds one term into an accumulator of the same type, scalar or vector.
  /// Both selections are inside the declared set, so the capability decides
  /// performance rather than legality.
  Value accumulateTerm(Location loc, Value lhs, Value rhs, Value accumulator,
                       OpBuilder &builder) const {
    if (fuseTerms)
      return ondrix::ondsp::consumeFastPermission(
          builder.create<math::FmaOp>(loc, lhs, rhs, accumulator),
          ondrix::ondsp::FastPermission::FuseMultiplyAdd);
    Value product = builder.create<arith::MulFOp>(loc, lhs, rhs);
    return builder.create<arith::AddFOp>(loc, accumulator, product);
  }

  Value createOrderedTail(Location loc, OpAdaptor adaptor, Value lowerBound, Value upperBound,
                          Value step, Value initial, OpBuilder &builder) const {
    auto loop = builder.create<scf::ForOp>(
        loc, lowerBound, upperBound, step, ValueRange{initial},
        [&](OpBuilder &bodyBuilder, Location bodyLoc, Value index, ValueRange iterArgs) {
          Value lhs = bodyBuilder.create<memref::LoadOp>(bodyLoc, adaptor.getLhs(), index);
          Value rhs = bodyBuilder.create<memref::LoadOp>(bodyLoc, adaptor.getRhs(), index);
          bodyBuilder.create<scf::YieldOp>(
              bodyLoc, accumulateTerm(bodyLoc, lhs, rhs, iterArgs.front(), bodyBuilder));
        });
    return loop.getResult(0);
  }

  int64_t vectorWidth;
  bool fuseTerms;
  int64_t interleave;
};

class VectorizeOndspFpFastMemRefReducePass final
    : public ondrix::impl::VectorizeOndspFpFastMemRefReduceBase<
          VectorizeOndspFpFastMemRefReducePass> {
public:
  using ondrix::impl::VectorizeOndspFpFastMemRefReduceBase<
      VectorizeOndspFpFastMemRefReducePass>::VectorizeOndspFpFastMemRefReduceBase;

  void runOnOperation() override {
    if (vectorWidth <= 1) {
      getOperation().emitError("vector-width must be greater than one");
      signalPassFailure();
      return;
    }
    if (vectorWidth > kMaxVectorWidth) {
      getOperation().emitError("vector-width must not exceed ") << kMaxVectorWidth;
      signalPassFailure();
      return;
    }

    if (interleave < 1 || interleave > kMaxInterleave) {
      getOperation().emitError("interleave must be in [1, ") << kMaxInterleave << "]";
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<FastReduceMacOpVectorization>(&getContext(), vectorWidth, supportsVectorFma,
                                               interleave);

    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, cf::ControlFlowDialect, math::MathDialect,
                           memref::MemRefDialect, ondrix::ondsp::OndspDialect, scf::SCFDialect,
                           vector::VectorDialect>();
    target.addDynamicallyLegalOp<ondrix::ondsp::ReduceMacOp>(
        [width = vectorWidth.getValue()](ondrix::ondsp::ReduceMacOp op) {
          return !isSupportedFastMemRefReduction(op, width);
        });

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      return signalPassFailure();
    ondrix::ondsp::summarizeFastPermissions(getOperation());
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createVectorizeOndspFpFastMemRefReducePass() {
  return std::make_unique<VectorizeOndspFpFastMemRefReducePass>();
}

std::unique_ptr<Pass> ondrix::createVectorizeOndspFpFastMemRefReducePass(
    const ondrix::VectorizeOndspFpFastMemRefReduceOptions &options) {
  return std::make_unique<VectorizeOndspFpFastMemRefReducePass>(options);
}

#include "ondrix/Dialect/ondrix/Transforms/BufferizableOpInterfaceImpl.h"

#include "ondrix/Dialect/ondrix/IR/OndrixDialect.h"
#include "ondrix/Dialect/ondrix/IR/OndrixOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace ondrix::ir {
namespace {

template <typename OpTy>
static Value createInitialAccumulator(OpTy op, OpBuilder &builder, Location loc) {
  if (isa<ondrix::ondsp::FixedAttr>(op.getNumeric()))
    return builder.create<ondrix::ondsp::AccZeroOp>(loc, *op.getAccumulator());
  auto fp = cast<ondrix::ondsp::FpAttr>(op.getNumeric());
  return builder.create<arith::ConstantOp>(loc, fp.getFormat(),
                                           builder.getZeroAttr(fp.getFormat()));
}

static Value createFpAccumulatorUpdate(Location loc, Value lhs, Value rhs, Value accumulator,
                                       ondrix::ondsp::FpAttr numeric, OpBuilder &builder) {
  switch (numeric.getContract()) {
  case ondrix::ondsp::FpContractMode::Off: {
    Value product = builder.create<arith::MulFOp>(loc, lhs, rhs);
    return builder.create<arith::AddFOp>(loc, accumulator, product);
  }
  case ondrix::ondsp::FpContractMode::Fma:
    return builder.create<math::FmaOp>(loc, lhs, rhs, accumulator);
  case ondrix::ondsp::FpContractMode::Fast:
    return builder.create<math::FmaOp>(loc, lhs, rhs, accumulator, arith::FastMathFlags::fast);
  }
  llvm_unreachable("unknown floating-point contract mode");
}

template <typename OpTy>
static Value exportFirSample(OpTy op, Value accumulator, OpBuilder &builder, Location loc) {
  if (!isa<ondrix::ondsp::FixedAttr>(op.getNumeric()))
    return accumulator;
  return builder.create<ondrix::ondsp::AccExportOp>(loc, op.getDst()->getStorage(), accumulator,
                                                    *op.getDst(), *op.getRounding(),
                                                    *op.getOverflow());
}

template <typename OpTy>
static Value createReducedFirSample(OpTy op, Value window, Value coefficients, OpBuilder &builder,
                                    Location loc) {
  Value initial = createInitialAccumulator(op, builder, loc);
  Value reduced = builder.create<ondrix::ondsp::ReduceMacOp>(
      loc, initial.getType(), initial, window, coefficients, op.getNumeric(),
      op.getProduct().value_or(ondrix::ondsp::ProductAttr()));
  return exportFirSample(op, reduced, builder, loc);
}

static Value createGuardedFullFirSample(FirFilterOp op, Value input, Value coefficients,
                                        Value outputIndex, Value inputLength,
                                        Value coefficientLength, Value leftPadding, Value zero,
                                        Value one, OpBuilder &builder, Location loc) {
  Value initial = createInitialAccumulator(op, builder, loc);
  auto fixed = dyn_cast<ondrix::ondsp::FixedAttr>(op.getNumeric());
  auto fp = dyn_cast<ondrix::ondsp::FpAttr>(op.getNumeric());
  Value outputBeforeLeft =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, outputIndex, leftPadding);
  Value leftDeficit = builder.create<arith::SubIOp>(loc, leftPadding, outputIndex);
  Value firstValidTap = builder.create<arith::SelectOp>(loc, outputBeforeLeft, leftDeficit, zero);
  Value inputBase = builder.create<arith::SubIOp>(loc, outputIndex, leftPadding);
  auto tapLoop = builder.create<scf::ForOp>(
      loc, zero, coefficientLength, one, ValueRange{initial},
      [&](OpBuilder &tapBuilder, Location tapLoc, Value tap, ValueRange accumulatorArgs) {
        Value pastLeftPadding =
            tapBuilder.create<arith::CmpIOp>(tapLoc, arith::CmpIPredicate::uge, tap, firstValidTap);
        Value inputIndex = tapBuilder.create<arith::AddIOp>(tapLoc, inputBase, tap);
        Value beforeRightPadding = tapBuilder.create<arith::CmpIOp>(
            tapLoc, arith::CmpIPredicate::ult, inputIndex, inputLength);
        Value inBounds =
            tapBuilder.create<arith::AndIOp>(tapLoc, pastLeftPadding, beforeRightPadding);
        auto guarded = tapBuilder.create<scf::IfOp>(
            tapLoc, TypeRange{accumulatorArgs.front().getType()}, inBounds,
            /*withElseRegion=*/true);

        OpBuilder thenBuilder = guarded.getThenBodyBuilder();
        Value inputValue = thenBuilder.create<memref::LoadOp>(tapLoc, input, inputIndex);
        Value coefficient = thenBuilder.create<memref::LoadOp>(tapLoc, coefficients, tap);
        Value updated;
        if (fixed) {
          updated = thenBuilder.create<ondrix::ondsp::MacOp>(tapLoc, *op.getAccumulator(),
                                                             accumulatorArgs.front(), inputValue,
                                                             coefficient, fixed, *op.getProduct());
        } else {
          updated = createFpAccumulatorUpdate(tapLoc, inputValue, coefficient,
                                              accumulatorArgs.front(), fp, thenBuilder);
        }
        thenBuilder.create<scf::YieldOp>(tapLoc, updated);

        OpBuilder elseBuilder = guarded.getElseBodyBuilder();
        elseBuilder.create<scf::YieldOp>(tapLoc, accumulatorArgs.front());
        tapBuilder.create<scf::YieldOp>(tapLoc, guarded.getResult(0));
      });
  return exportFirSample(op, tapLoop.getResult(0), builder, loc);
}

static void assertFullFirFilterShape(Location loc, Value inputLength, Value coefficientLength,
                                     Value outputLength, Value zero, Value one,
                                     OpBuilder &builder) {
  Value hasInput = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, inputLength, zero);
  builder.create<cf::AssertOp>(
      loc, hasInput, builder.getStringAttr("full FIR requires at least one input sample"));
  Value hasCoefficients =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, coefficientLength, zero);
  builder.create<cf::AssertOp>(loc, hasCoefficients,
                               builder.getStringAttr("full FIR requires at least one coefficient"));

  Value leftPadding = builder.create<arith::SubIOp>(loc, coefficientLength, one);
  Value outputCoversPadding =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge, outputLength, leftPadding);
  Value recoveredInput = builder.create<arith::SubIOp>(loc, outputLength, leftPadding);
  Value outputMatches =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, recoveredInput, inputLength);
  Value validOutputShape = builder.create<arith::AndIOp>(loc, outputCoversPadding, outputMatches);
  builder.create<cf::AssertOp>(
      loc, validOutputShape,
      builder.getStringAttr(
          "full FIR output length must equal input length plus coefficient length minus one"));
}

static void assertFullFirFilterTileShape(Location loc, Value inputLength, Value coefficientLength,
                                         Value outputLength, Value outputOrigin, Value zero,
                                         Value one, OpBuilder &builder) {
  Value hasInput = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, inputLength, zero);
  builder.create<cf::AssertOp>(
      loc, hasInput, builder.getStringAttr("full FIR requires at least one input sample"));
  Value hasCoefficients =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, coefficientLength, zero);
  builder.create<cf::AssertOp>(loc, hasCoefficients,
                               builder.getStringAttr("full FIR requires at least one coefficient"));

  Value leftPadding = builder.create<arith::SubIOp>(loc, coefficientLength, one);
  Value completeOutputLength = builder.create<arith::AddIOp>(loc, inputLength, leftPadding);
  Value extentDidNotOverflow = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge,
                                                             completeOutputLength, inputLength);
  Value originInRange = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ule, outputOrigin,
                                                      completeOutputLength);
  Value remaining = builder.create<arith::SubIOp>(loc, completeOutputLength, outputOrigin);
  Value tileInRange =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ule, outputLength, remaining);
  Value validRange = builder.create<arith::AndIOp>(loc, extentDidNotOverflow, originInRange);
  validRange = builder.create<arith::AndIOp>(loc, validRange, tileInRange);
  builder.create<cf::AssertOp>(
      loc, validRange,
      builder.getStringAttr("full FIR output tile must lie within the complete output range"));
}

static void assertValidConv1DShape(Location loc, Value inputLength, Value kernelLength,
                                   Value outputLength, Value zero, Value one, OpBuilder &builder) {
  Value hasKernel =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, kernelLength, zero);
  builder.create<cf::AssertOp>(
      loc, hasKernel, builder.getStringAttr("conv1d requires at least one kernel element"));
  Value inputCoversKernel =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge, inputLength, kernelLength);
  builder.create<cf::AssertOp>(
      loc, inputCoversKernel,
      builder.getStringAttr("conv1d input must cover one complete kernel window"));
  Value remaining = builder.create<arith::SubIOp>(loc, inputLength, kernelLength);
  Value requiredOutputLength = builder.create<arith::AddIOp>(loc, remaining, one);
  Value outputMatches = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, outputLength,
                                                      requiredOutputLength);
  builder.create<cf::AssertOp>(
      loc, outputMatches,
      builder.getStringAttr(
          "conv1d output length must equal input length minus kernel length plus one"));
}

static void assertValidFirDecimateShape(Location loc, Value inputLength, Value coefficientLength,
                                        Value outputLength, Value factor, Value zero, Value one,
                                        OpBuilder &builder) {
  Value hasCoefficients =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, coefficientLength, zero);
  builder.create<cf::AssertOp>(
      loc, hasCoefficients,
      builder.getStringAttr("fir_decimate requires at least one coefficient"));
  Value inputCoversCoefficients =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge, inputLength, coefficientLength);
  builder.create<cf::AssertOp>(
      loc, inputCoversCoefficients,
      builder.getStringAttr("fir_decimate input must cover one complete coefficient window"));
  Value remaining = builder.create<arith::SubIOp>(loc, inputLength, coefficientLength);
  Value completeSteps = builder.create<arith::DivUIOp>(loc, remaining, factor);
  Value requiredOutputLength = builder.create<arith::AddIOp>(loc, completeSteps, one);
  Value outputMatches = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, outputLength,
                                                      requiredOutputLength);
  builder.create<cf::AssertOp>(
      loc, outputMatches,
      builder.getStringAttr(
          "fir_decimate output length must equal floor((input length - coefficient length) / "
          "factor) plus one"));
}

static Value createReducedFirDecimateSample(FirDecimateOp op, Value window, Value coefficients,
                                            OpBuilder &builder, Location loc) {
  Value initial = builder.create<ondrix::ondsp::AccZeroOp>(loc, op.getAccumulator());
  Value reduced = builder.create<ondrix::ondsp::ReduceMacOp>(
      loc, initial.getType(), initial, window, coefficients, op.getNumeric(), op.getProduct());
  return builder.create<ondrix::ondsp::AccExportOp>(
      loc, op.getDst().getStorage(), reduced, op.getDst(), op.getRounding(), op.getOverflow());
}

struct FirFilterOpInterface
    : public DstBufferizableOpInterfaceExternalModel<FirFilterOpInterface, FirFilterOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand, const AnalysisState &) const {
    auto fir = cast<FirFilterOp>(op);
    return !fir.isDpsInit(&opOperand);
  }

  LogicalResult bufferize(Operation *operation, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto op = cast<FirFilterOp>(operation);
    FailureOr<Value> input = getBuffer(rewriter, op.getInput(), options);
    FailureOr<Value> coefficients = getBuffer(rewriter, op.getCoeffs(), options);
    FailureOr<Value> output = getBuffer(rewriter, op.getInit(), options);
    if (failed(input) || failed(coefficients) || failed(output))
      return failure();

    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value inputLength = rewriter.create<memref::DimOp>(loc, *input, zero);
    Value coefficientLength = rewriter.create<memref::DimOp>(loc, *coefficients, zero);
    Value outputLength = rewriter.create<memref::DimOp>(loc, *output, zero);
    Value globalOutputOrigin = op.getOutputOrigin() ? op.getOutputOrigin() : zero;
    SmallVector<OpFoldResult> coefficientOffsets{rewriter.getIndexAttr(0)};
    SmallVector<OpFoldResult> coefficientSizes{coefficientLength};
    SmallVector<OpFoldResult> coefficientStrides{rewriter.getIndexAttr(1)};
    Value coefficientView = rewriter.create<memref::SubViewOp>(
        loc, *coefficients, coefficientOffsets, coefficientSizes, coefficientStrides);

    auto createValidRange = [&](Value lower, Value upper) {
      rewriter.create<scf::ForOp>(
          loc, lower, upper, one, ValueRange{},
          [&](OpBuilder &builder, Location bodyLoc, Value localOutputIndex, ValueRange) {
            Value globalOutputIndex = localOutputIndex;
            if (op.getOutputOrigin())
              globalOutputIndex =
                  builder.create<arith::AddIOp>(bodyLoc, globalOutputOrigin, localOutputIndex);
            Value inputOffset = globalOutputIndex;
            if (op.getBoundary() == FirBoundaryMode::Full) {
              Value leftPadding = builder.create<arith::SubIOp>(bodyLoc, coefficientLength, one);
              inputOffset = builder.create<arith::SubIOp>(bodyLoc, globalOutputIndex, leftPadding);
            }
            SmallVector<OpFoldResult> offsets{inputOffset};
            SmallVector<OpFoldResult> sizes{coefficientLength};
            SmallVector<OpFoldResult> strides{builder.getIndexAttr(1)};
            Value window =
                builder.create<memref::SubViewOp>(bodyLoc, *input, offsets, sizes, strides);
            Value sample = createReducedFirSample(op, window, coefficientView, builder, bodyLoc);
            builder.create<memref::StoreOp>(bodyLoc, sample, *output, localOutputIndex);
            builder.create<scf::YieldOp>(bodyLoc);
          });
    };

    if (op.getBoundary() == FirBoundaryMode::Valid) {
      createValidRange(zero, outputLength);
    } else {
      if (op.getOutputOrigin())
        assertFullFirFilterTileShape(loc, inputLength, coefficientLength, outputLength,
                                     globalOutputOrigin, zero, one, rewriter);
      else
        assertFullFirFilterShape(loc, inputLength, coefficientLength, outputLength, zero, one,
                                 rewriter);
      Value leftPadding = rewriter.create<arith::SubIOp>(loc, coefficientLength, one);
      auto createGuardedRange = [&](Value lower, Value upper) {
        rewriter.create<scf::ForOp>(
            loc, lower, upper, one, ValueRange{},
            [&](OpBuilder &builder, Location bodyLoc, Value localOutputIndex, ValueRange) {
              Value globalOutputIndex = localOutputIndex;
              if (op.getOutputOrigin())
                globalOutputIndex =
                    builder.create<arith::AddIOp>(bodyLoc, globalOutputOrigin, localOutputIndex);
              Value sample = createGuardedFullFirSample(
                  op, *input, coefficientView, globalOutputIndex, inputLength, coefficientLength,
                  leftPadding, zero, one, builder, bodyLoc);
              builder.create<memref::StoreOp>(bodyLoc, sample, *output, localOutputIndex);
              builder.create<scf::YieldOp>(bodyLoc);
            });
      };

      Value globalOutputEnd = outputLength;
      if (op.getOutputOrigin())
        globalOutputEnd = rewriter.create<arith::AddIOp>(loc, globalOutputOrigin, outputLength);

      Value leftEnd = rewriter.create<arith::MinUIOp>(loc, globalOutputEnd, leftPadding);
      leftEnd = rewriter.create<arith::MaxUIOp>(loc, leftEnd, globalOutputOrigin);
      Value localLeftEnd = rewriter.create<arith::SubIOp>(loc, leftEnd, globalOutputOrigin);
      createGuardedRange(zero, localLeftEnd);

      Value interiorStart = rewriter.create<arith::MaxUIOp>(loc, globalOutputOrigin, leftPadding);
      Value interiorEnd = rewriter.create<arith::MinUIOp>(loc, globalOutputEnd, inputLength);
      interiorEnd = rewriter.create<arith::MaxUIOp>(loc, interiorEnd, interiorStart);
      Value localInteriorStart =
          rewriter.create<arith::SubIOp>(loc, interiorStart, globalOutputOrigin);
      Value localInteriorEnd = rewriter.create<arith::SubIOp>(loc, interiorEnd, globalOutputOrigin);
      createValidRange(localInteriorStart, localInteriorEnd);

      Value rightBoundary = rewriter.create<arith::MaxUIOp>(loc, inputLength, leftPadding);
      Value rightStart = rewriter.create<arith::MaxUIOp>(loc, globalOutputOrigin, rightBoundary);
      rightStart = rewriter.create<arith::MinUIOp>(loc, rightStart, globalOutputEnd);
      Value localRightStart = rewriter.create<arith::SubIOp>(loc, rightStart, globalOutputOrigin);
      createGuardedRange(localRightStart, outputLength);
    }

    replaceOpWithBufferizedValues(rewriter, op, *output);
    return success();
  }
};

struct FirDecimateOpInterface
    : public DstBufferizableOpInterfaceExternalModel<FirDecimateOpInterface, FirDecimateOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand, const AnalysisState &) const {
    auto decimate = cast<FirDecimateOp>(op);
    return !decimate.isDpsInit(&opOperand);
  }

  LogicalResult bufferize(Operation *operation, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto op = cast<FirDecimateOp>(operation);
    FailureOr<Value> input = getBuffer(rewriter, op.getInput(), options);
    FailureOr<Value> coefficients = getBuffer(rewriter, op.getCoeffs(), options);
    FailureOr<Value> output = getBuffer(rewriter, op.getInit(), options);
    if (failed(input) || failed(coefficients) || failed(output))
      return failure();

    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value factor =
        rewriter.create<arith::ConstantIndexOp>(loc, op.getFactorAttr().getValue().getSExtValue());
    Value inputLength = rewriter.create<memref::DimOp>(loc, *input, zero);
    Value coefficientLength = rewriter.create<memref::DimOp>(loc, *coefficients, zero);
    Value outputLength = rewriter.create<memref::DimOp>(loc, *output, zero);
    assertValidFirDecimateShape(loc, inputLength, coefficientLength, outputLength, factor, zero,
                                one, rewriter);

    Value coefficientView = rewriter.create<memref::SubViewOp>(
        loc, *coefficients, ArrayRef<OpFoldResult>{rewriter.getIndexAttr(0)},
        ArrayRef<OpFoldResult>{coefficientLength},
        ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)});
    rewriter.create<scf::ForOp>(
        loc, zero, outputLength, one, ValueRange{},
        [&](OpBuilder &builder, Location bodyLoc, Value outputIndex, ValueRange) {
          Value inputOffset = builder.create<arith::MulIOp>(bodyLoc, outputIndex, factor);
          Value inputWindow = builder.create<memref::SubViewOp>(
              bodyLoc, *input, ArrayRef<OpFoldResult>{inputOffset},
              ArrayRef<OpFoldResult>{coefficientLength},
              ArrayRef<OpFoldResult>{builder.getIndexAttr(1)});
          Value sample =
              createReducedFirDecimateSample(op, inputWindow, coefficientView, builder, bodyLoc);
          builder.create<memref::StoreOp>(bodyLoc, sample, *output, outputIndex);
          builder.create<scf::YieldOp>(bodyLoc);
        });

    replaceOpWithBufferizedValues(rewriter, op, *output);
    return success();
  }
};

/// Allocate the result buffer of a value-producing algorithm operation. The
/// framework owns the escape/deallocation decision; the caller only fills the
/// returned buffer.
static FailureOr<Value> createProducedResultBuffer(RewriterBase &rewriter, Value tensorResult,
                                                   const BufferizationOptions &options) {
  auto tensorType = cast<RankedTensorType>(tensorResult.getType());
  Location loc = tensorResult.getLoc();
  bool dealloc = shouldDeallocateOpResult(cast<OpResult>(tensorResult), options);
  FailureOr<Value> allocated = allocateTensorForShapedValue(rewriter, loc, tensorResult,
                                                            /*escape=*/!dealloc, options,
                                                            /*copy=*/false);
  if (failed(allocated))
    return failure();
  auto memrefType = MemRefType::get(tensorType.getShape(), tensorType.getElementType());
  return rewriter.create<bufferization::ToMemrefOp>(loc, memrefType, *allocated).getResult();
}

/// Signed frac-30 wrapping accumulator of the requested width. Wrap is the
/// exact-modulo reassociation class, so a reduction seeded at zero with a
/// provably non-wrapping range is reassociable without a prefix proof.
static ondrix::ondsp::AccType getExactWrapAccumulator(MLIRContext *context, unsigned width) {
  return ondrix::ondsp::AccType::get(context, IntegerType::get(context, width), /*frac=*/30,
                                     ondrix::ondsp::Signedness::Signed,
                                     ondrix::ondsp::OverflowMode::Wrap);
}

/// Rank-reduced unit-stride view of one row of a rank-2 memref.
static Value createUnitStrideRowView(OpBuilder &builder, Location loc, Value matrix, Value row,
                                     int64_t length) {
  SmallVector<OpFoldResult> offsets{row, builder.getIndexAttr(0)};
  SmallVector<OpFoldResult> sizes{builder.getIndexAttr(1), builder.getIndexAttr(length)};
  SmallVector<OpFoldResult> strides{builder.getIndexAttr(1), builder.getIndexAttr(1)};
  auto viewType = cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
      {length}, cast<MemRefType>(matrix.getType()), offsets, sizes, strides));
  return builder.create<memref::SubViewOp>(loc, viewType, matrix, offsets, sizes, strides);
}

struct MatmulOpInterface
    : public BufferizableOpInterface::ExternalModel<MatmulOpInterface, MatmulOp> {
  bool bufferizesToAllocation(Operation *, OpResult) const { return true; }

  bool bufferizesToMemoryRead(Operation *, OpOperand &, const AnalysisState &) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *, OpOperand &, const AnalysisState &) const {
    return false;
  }

  AliasingOpResultList getAliasingOpResults(Operation *, OpOperand &, const AnalysisState &) const {
    return {};
  }

  LogicalResult bufferize(Operation *operation, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto op = cast<MatmulOp>(operation);
    FailureOr<Value> lhs = getBuffer(rewriter, op.getLhs(), options);
    FailureOr<Value> rhs = getBuffer(rewriter, op.getRhs(), options);
    if (failed(lhs) || failed(rhs))
      return failure();

    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();
    FailureOr<Value> output = createProducedResultBuffer(rewriter, op.getResult(), options);
    if (failed(output))
      return failure();

    RankedTensorType lhsType = op.getLhs().getType();
    RankedTensorType rhsType = op.getRhs().getType();
    int64_t rowCount = lhsType.getDimSize(0);
    int64_t innerCount = lhsType.getDimSize(1);
    int64_t columnCount = rhsType.getDimSize(1);
    auto elementType = cast<IntegerType>(lhsType.getElementType());
    auto numeric = cast<ondrix::ondsp::FixedAttr>(op.getNumeric());
    auto product = ondrix::ondsp::ProductAttr::get(context, ondrix::ondsp::ProductSelection::Full);
    // Two independent arguments meet here. Reassociation legality needs only
    // the wrap overflow mode: exact-modulo accumulation is associative at any
    // width, so the horizontal Vector consumer may reassociate with no
    // constant-coefficient or prefix-range proof and no range assumption.
    // The range bound is what ties the wrapped value to the CONTRACT: every
    // full product magnitude is at most 2^30 and K is at most 64, so
    // |sum| <= 64 * 2^30 = 2^36 < 2^39 and the i40 wrapping accumulator never
    // actually wraps — its value equals the exact K-sum of the tensor-form
    // lowering.
    ondrix::ondsp::AccType accumulatorType = getExactWrapAccumulator(context, /*width=*/40);

    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value rows = rewriter.create<arith::ConstantIndexOp>(loc, rowCount);
    Value inner = rewriter.create<arith::ConstantIndexOp>(loc, innerCount);
    Value columns = rewriter.create<arith::ConstantIndexOp>(loc, columnCount);

    // The columns of B have stride N and would be refused by the unit-stride
    // Vector legality gate. Pack B once into a transposed scratch buffer so
    // the packed operand is a unit-stride rank-1 view; the pack is pure data
    // movement and crosses no numeric boundary. The A-row view below inherits
    // the layout of the incoming buffer, so it is unit-stride only under
    // identity-layout function boundaries — under fully dynamic layouts the
    // Vector legality gate refuses both views and the reduction stays in its
    // ordered scalar form, which is a performance fallback, never a semantic
    // change.
    auto packedType = MemRefType::get({columnCount, innerCount}, elementType);
    FailureOr<Value> packed = options.createAlloc(rewriter, loc, packedType, /*dynShape=*/{});
    if (failed(packed))
      return failure();
    rewriter.create<scf::ForOp>(
        loc, zero, columns, one, ValueRange{},
        [&](OpBuilder &builder, Location columnLoc, Value column, ValueRange) {
          builder.create<scf::ForOp>(
              columnLoc, zero, inner, one, ValueRange{},
              [&](OpBuilder &innerBuilder, Location innerLoc, Value index, ValueRange) {
                Value element =
                    innerBuilder.create<memref::LoadOp>(innerLoc, *rhs, ValueRange{index, column});
                innerBuilder.create<memref::StoreOp>(innerLoc, element, *packed,
                                                     ValueRange{column, index});
                innerBuilder.create<scf::YieldOp>(innerLoc);
              });
          builder.create<scf::YieldOp>(columnLoc);
        });

    rewriter.create<scf::ForOp>(
        loc, zero, rows, one, ValueRange{},
        [&](OpBuilder &builder, Location rowLoc, Value row, ValueRange) {
          Value lhsRow = createUnitStrideRowView(builder, rowLoc, *lhs, row, innerCount);
          builder.create<scf::ForOp>(
              rowLoc, zero, columns, one, ValueRange{},
              [&](OpBuilder &columnBuilder, Location columnLoc, Value column, ValueRange) {
                Value packedRow =
                    createUnitStrideRowView(columnBuilder, columnLoc, *packed, column, innerCount);
                Value initial =
                    columnBuilder.create<ondrix::ondsp::AccZeroOp>(columnLoc, accumulatorType);
                Value reduced = columnBuilder.create<ondrix::ondsp::ReduceMacOp>(
                    columnLoc, accumulatorType, initial, lhsRow, packedRow, numeric, product);
                // Dividing the raw accumulator by 2^(30 - 15) with nearest-even
                // rounding and saturating to i16 is exactly the `round_shift`
                // boundary of the tensor-form lowering.
                Value element = columnBuilder.create<ondrix::ondsp::AccExportOp>(
                    columnLoc, elementType, reduced, numeric, op.getRounding(),
                    ondrix::ondsp::OverflowMode::Saturate);
                columnBuilder.create<memref::StoreOp>(columnLoc, element, *output,
                                                      ValueRange{row, column});
                columnBuilder.create<scf::YieldOp>(columnLoc);
              });
          builder.create<scf::YieldOp>(rowLoc);
        });

    if (failed(options.createDealloc(rewriter, loc, *packed)))
      return failure();
    replaceOpWithBufferizedValues(rewriter, op, *output);
    return success();
  }
};

struct RmsOpInterface : public BufferizableOpInterface::ExternalModel<RmsOpInterface, RmsOp> {
  bool bufferizesToAllocation(Operation *, OpResult) const { return true; }

  bool bufferizesToMemoryRead(Operation *, OpOperand &, const AnalysisState &) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *, OpOperand &, const AnalysisState &) const {
    return false;
  }

  AliasingOpResultList getAliasingOpResults(Operation *, OpOperand &, const AnalysisState &) const {
    return {};
  }

  LogicalResult bufferize(Operation *operation, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto op = cast<RmsOp>(operation);
    FailureOr<Value> input = getBuffer(rewriter, op.getInput(), options);
    if (failed(input))
      return failure();

    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();
    FailureOr<Value> output = createProducedResultBuffer(rewriter, op.getResult(), options);
    if (failed(output))
      return failure();

    int64_t extent = op.getInput().getType().getDimSize(0);
    unsigned meanShift = llvm::Log2_64(extent);
    IntegerType i32 = rewriter.getIntegerType(32);
    IntegerType i64 = rewriter.getIntegerType(64);
    auto numeric = cast<ondrix::ondsp::FixedAttr>(op.getNumeric());
    auto product = ondrix::ondsp::ProductAttr::get(context, ondrix::ondsp::ProductSelection::Full);
    // Squaring the input is one reduction whose two operands are the same
    // buffer. As with matmul, wrap alone authorizes reassociation; the range
    // bound is what ties the wrapped value to the contract: every square is
    // at most 2^30 and N is at most 4096, so the sum is at most 2^42 < 2^63
    // and the i64 wrapping accumulator never actually wraps — it equals the
    // exact sum of squares. An i40 accumulator would NOT suffice
    // (2^42 > 2^39), which is why this reduction needs the wider wrapping
    // accumulator admitted by the horizontal-domain predicate.
    ondrix::ondsp::AccType accumulatorType = getExactWrapAccumulator(context, /*width=*/64);
    Value initial = rewriter.create<ondrix::ondsp::AccZeroOp>(loc, accumulatorType);
    Value reduced = rewriter.create<ondrix::ondsp::ReduceMacOp>(loc, accumulatorType, initial,
                                                                *input, *input, numeric, product);
    // Materialize the exact raw sum at its own reading (identity export at
    // frac 30), then apply the nearest-even saturating mean by 2^m as a
    // declared ARITHMETIC `round_shift` — the same boundary op the
    // tensor-form lowering uses. `acc_export`'s destination frac is a
    // value-preserving reading, never a shift selector; the mean changes
    // the represented value and therefore must not be expressed through
    // it. The declared i32 saturation of the mean is unreachable because
    // the mean of squares is at most 2^30.
    auto sumFormat = ondrix::ondsp::FixedAttr::get(context, ondrix::ondsp::Signedness::Signed, i64,
                                                   /*frac=*/30);
    Value sum = rewriter.create<ondrix::ondsp::AccExportOp>(
        loc, i64, reduced, sumFormat, ondrix::ondsp::RoundingMode::NearestEven,
        ondrix::ondsp::OverflowMode::Saturate);
    auto meanScale = ondrix::ondsp::ScaleAttr::get(
        context, /*preShiftLeft=*/0, /*postShiftRight=*/meanShift,
        ondrix::ondsp::RoundingMode::NearestEven, ondrix::ondsp::OverflowMode::Saturate, i32);
    Value mean = rewriter.create<ondrix::ondsp::RoundShiftOp>(loc, i32, sum, meanScale);
    Value meanWide = rewriter.create<arith::ExtSIOp>(loc, i64, mean);
    Value root = rewriter.create<ondrix::ondsp::SqrtFixedOp>(loc, rewriter.getI16Type(), meanWide,
                                                             op.getRoundingAttr());
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    rewriter.create<memref::StoreOp>(loc, root, *output, ValueRange{zero});

    replaceOpWithBufferizedValues(rewriter, op, *output);
    return success();
  }
};

struct Conv1DOpInterface
    : public DstBufferizableOpInterfaceExternalModel<Conv1DOpInterface, Conv1DOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand, const AnalysisState &) const {
    auto conv = cast<Conv1DOp>(op);
    return !conv.isDpsInit(&opOperand);
  }

  LogicalResult bufferize(Operation *operation, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto op = cast<Conv1DOp>(operation);
    FailureOr<Value> input = getBuffer(rewriter, op.getInput(), options);
    FailureOr<Value> kernel = getBuffer(rewriter, op.getKernel(), options);
    FailureOr<Value> output = getBuffer(rewriter, op.getInit(), options);
    if (failed(input) || failed(kernel) || failed(output))
      return failure();

    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value inputLength = rewriter.create<memref::DimOp>(loc, *input, zero);
    Value kernelLength = rewriter.create<memref::DimOp>(loc, *kernel, zero);
    Value outputLength = rewriter.create<memref::DimOp>(loc, *output, zero);
    assertValidConv1DShape(loc, inputLength, kernelLength, outputLength, zero, one, rewriter);

    OpFoldResult kernelOffset = rewriter.getIndexAttr(0);
    OpFoldResult kernelStride = rewriter.getIndexAttr(1);
    if (op.getMode() == Conv1DMode::Convolution) {
      kernelOffset = rewriter.create<arith::SubIOp>(loc, kernelLength, one).getResult();
      kernelStride = rewriter.getIndexAttr(-1);
    }
    Value kernelView = rewriter.create<memref::SubViewOp>(
        loc, *kernel, ArrayRef<OpFoldResult>{kernelOffset}, ArrayRef<OpFoldResult>{kernelLength},
        ArrayRef<OpFoldResult>{kernelStride});

    rewriter.create<scf::ForOp>(
        loc, zero, outputLength, one, ValueRange{},
        [&](OpBuilder &builder, Location bodyLoc, Value outputIndex, ValueRange) {
          Value inputWindow = builder.create<memref::SubViewOp>(
              bodyLoc, *input, ArrayRef<OpFoldResult>{outputIndex},
              ArrayRef<OpFoldResult>{kernelLength},
              ArrayRef<OpFoldResult>{builder.getIndexAttr(1)});
          Value sample = createReducedFirSample(op, inputWindow, kernelView, builder, bodyLoc);
          builder.create<memref::StoreOp>(bodyLoc, sample, *output, outputIndex);
          builder.create<scf::YieldOp>(bodyLoc);
        });

    replaceOpWithBufferizedValues(rewriter, op, *output);
    return success();
  }
};

} // namespace

void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, OndrixDialect *) {
    FirFilterOp::attachInterface<FirFilterOpInterface>(*context);
    FirDecimateOp::attachInterface<FirDecimateOpInterface>(*context);
    Conv1DOp::attachInterface<Conv1DOpInterface>(*context);
    MatmulOp::attachInterface<MatmulOpInterface>(*context);
    RmsOp::attachInterface<RmsOpInterface>(*context);

    // Bufferization materializes these dialects even when the input module
    // contains only tensor-form Ondrix operations.
    context->loadDialect<arith::ArithDialect, cf::ControlFlowDialect, math::MathDialect,
                         memref::MemRefDialect, scf::SCFDialect, ondrix::ondsp::OndspDialect>();
  });
}

} // namespace ondrix::ir

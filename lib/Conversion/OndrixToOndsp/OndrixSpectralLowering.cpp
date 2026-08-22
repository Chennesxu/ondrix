#include "OndrixToOndspCommon.h"

#include "ondrix/Dialect/ondrix/IR/OndrixDialect.h"
#include "ondrix/Dialect/ondrix/IR/OndrixOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"
#include "ondrix/Support/DctCoefficients.h"
#include "ondrix/Support/GuardedQ15Quantization.h"
#include "ondrix/Support/Q30SplitTwiddleTables.h"
#include "ondrix/Support/Q31TwiddleTables.h"

#include "llvm/ADT/APInt.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <optional>

using namespace mlir;
using namespace ondrix::conversion;

namespace {

class ButterflyOpLowering final : public OpConversionPattern<ondrix::ir::ButterflyOp> {
public:
  using OpConversionPattern<ondrix::ir::ButterflyOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ir::ButterflyOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto layout = dyn_cast<ondrix::ondsp::CxLayoutAttr>(op.getLayout());
    if (!layout)
      return rewriter.notifyMatchFailure(op, "requires an ondsp.cx_layout layout attribute");

    auto replacement = rewriter.create<ondrix::ondsp::CxButterflyOp>(
        op.getLoc(), op.getOut0().getType(), op.getOut1().getType(), adaptor.getA(), adaptor.getB(),
        adaptor.getTwiddle(), layout, op.getNumeric(), op.getProduct(), op.getProductScale(),
        op.getOutputScale(), ondrix::ondsp::CxButterflyVariantAttr());
    rewriter.replaceOp(op, replacement);
    return success();
  }
};

// Guard argument in GuardedQ15Quantization.h. +1.0 saturates to 32767 by
// declared convention; -1.0 is exact. A 50-digit sweep of every stage
// twiddle component for power-of-two sizes up to 1024 shows a worst-case
// margin of 0.0036 LSB, so all supported extents are admissible; the guard
// remains as the fail-closed backstop.
static std::optional<int64_t> quantizeTwiddleComponentQ15(double value) {
  std::optional<ondrix::GuardedQ15Value> quantized = ondrix::quantizeGuardedQ15(value);
  if (!quantized)
    return std::nullopt;
  return quantized->value;
}

static std::optional<uint64_t> getPackedQ15TwiddleBits(ondrix::ir::CfftDirection direction,
                                                       int64_t size, int64_t index) {
  constexpr double kTwoPi = 6.28318530717958647692528676655900577;
  double angle = kTwoPi * static_cast<double>(index) / static_cast<double>(size);
  std::optional<int64_t> real = quantizeTwiddleComponentQ15(std::cos(angle));
  double sine = std::sin(angle);
  std::optional<int64_t> imaginary =
      quantizeTwiddleComponentQ15(direction == ondrix::ir::CfftDirection::Forward ? -sine : sine);
  if (!real || !imaginary)
    return std::nullopt;
  return (static_cast<uint64_t>(*imaginary & 0xFFFF) << 16) | static_cast<uint64_t>(*real & 0xFFFF);
}

// One stage twiddle of whichever profile the layout selects. Q15 quantizes a
// binary64 estimate under the tie guard; Q31 reads the offline-frozen table,
// because at Q31 the guard is no longer wide enough to certify an
// in-compiler estimate (see include/ondrix/Support/Q31TwiddleTables.h).
static std::optional<uint64_t> getPackedTwiddleBits(unsigned storageWidth,
                                                    ondrix::ir::CfftDirection direction,
                                                    int64_t size, int64_t index) {
  if (storageWidth == 16)
    return getPackedQ15TwiddleBits(direction, size, index);
  return ondrix::getPackedQ31TwiddleBits(direction == ondrix::ir::CfftDirection::Forward
                                             ? ondrix::Q31TwiddleDirection::Forward
                                             : ondrix::Q31TwiddleDirection::Inverse,
                                         size, index);
}

// Fail-closed availability of every stage twiddle needed by the recursive
// combine of one static extent. The recursion itself may then rely on
// twiddle generation succeeding.
static bool hasAdmissiblePackedTwiddleTables(unsigned storageWidth,
                                             ondrix::ir::CfftDirection direction, int64_t extent) {
  for (int64_t size = 2; size <= extent; size *= 2)
    for (int64_t index = 0; index < size / 2; ++index)
      if (!getPackedTwiddleBits(storageWidth, direction, size, index))
        return false;
  return true;
}

// The real-spectrum verifiers admit only layouts with an executable packed
// profile, so the lowering reads the widths from the op's own layout.
static ondrix::ondsp::PackedComplexProfile getVerifiedPackedProfile(Attribute layout) {
  std::optional<ondrix::ondsp::PackedComplexProfile> profile =
      ondrix::ondsp::getPackedComplexProfile(cast<ondrix::ondsp::CxLayoutAttr>(layout).getLayout());
  assert(profile && "the verified layout must have an executable profile");
  return *profile;
}

static bool isInventoryRounding(ondrix::ondsp::RoundingMode rounding) {
  return rounding == ondrix::ondsp::RoundingMode::TowardNegative ||
         rounding == ondrix::ondsp::RoundingMode::NearestTiesPositive;
}

static SmallVector<Value>
lowerPackedCfft(Location loc, ArrayRef<Value> inputs, ondrix::ir::CfftDirection direction,
                ondrix::ondsp::PackedComplexProfile profile, ondrix::ondsp::CxLayoutAttr layout,
                Attribute numeric, ondrix::ondsp::ProductAttr product,
                ondrix::ondsp::ScaleAttr productScale, ondrix::ondsp::ScaleAttr outputScale,
                bool vectorizeStaticCfft, ConversionPatternRewriter &rewriter) {
  IntegerType container = rewriter.getIntegerType(profile.containerWidth);
  auto createPackedTwiddle = [&](uint64_t bits) {
    return rewriter.create<arith::ConstantOp>(
        loc, container,
        rewriter.getIntegerAttr(container, llvm::APInt(profile.containerWidth, bits)));
  };
  auto createButterfly = [&](Value a, Value b, Value twiddle,
                             ondrix::ondsp::CxButterflyVariant variant =
                                 ondrix::ondsp::CxButterflyVariant::Plain) {
    auto attr = variant == ondrix::ondsp::CxButterflyVariant::Plain
                    ? ondrix::ondsp::CxButterflyVariantAttr()
                    : ondrix::ondsp::CxButterflyVariantAttr::get(rewriter.getContext(), variant);
    return rewriter.create<ondrix::ondsp::CxButterflyOp>(loc, container, container, a, b, twiddle,
                                                         layout, numeric, product, productScale,
                                                         outputScale, attr);
  };
  // Target-inventory profiles use the target equation (decisions 2026-08-17):
  // combines of size 2 and 4 are the exact unit forms with no product stage;
  // larger combines pair legs j and j + n/4 on the shared twiddle W(n, j)
  // with the second leg as the cross combine. -j never multiplies.
  bool inventoryPaired = profile.storageWidth == 16 &&
                         isInventoryRounding(productScale.getRounding()) &&
                         isInventoryRounding(outputScale.getRounding());
  auto buildVector = [&](ArrayRef<Value> values) {
    assert(!values.empty() && "CFFT stage vector must contain at least one lane");
    auto vectorType =
        VectorType::get({static_cast<int64_t>(values.size())}, values.front().getType());
    Value vector = rewriter.create<vector::BroadcastOp>(loc, vectorType, values.front());
    for (auto [index, value] : llvm::enumerate(values.drop_front()))
      vector = rewriter.create<vector::InsertOp>(
          loc, value, vector, ArrayRef<int64_t>{static_cast<int64_t>(index + 1)});
    return vector;
  };

  std::function<SmallVector<Value>(ArrayRef<Value>)> lowerCfft =
      [&](ArrayRef<Value> values) -> SmallVector<Value> {
    if (values.size() == 1)
      return {values.front()};

    SmallVector<Value> evenInputs;
    SmallVector<Value> oddInputs;
    evenInputs.reserve(values.size() / 2);
    oddInputs.reserve(values.size() / 2);
    for (auto [index, value] : llvm::enumerate(values))
      (index % 2 == 0 ? evenInputs : oddInputs).push_back(value);

    SmallVector<Value> even = lowerCfft(evenInputs);
    SmallVector<Value> odd = lowerCfft(oddInputs);
    SmallVector<Value> outputs(values.size());
    if (inventoryPaired) {
      int64_t half = values.size() / 2;
      bool exact = values.size() <= 4;
      if (values.size() == 2) {
        auto unitLeg = createButterfly(even[0], odd[0], createPackedTwiddle(0x7FFF),
                                       ondrix::ondsp::CxButterflyVariant::Unit);
        outputs[0] = unitLeg.getOut0();
        outputs[1] = unitLeg.getOut1();
        return outputs;
      }
      for (int64_t index = 0; index < half / 2; ++index) {
        Value twiddle;
        if (exact) {
          twiddle = createPackedTwiddle(0x7FFF);
        } else {
          std::optional<uint64_t> twiddleBits =
              getPackedTwiddleBits(profile.storageWidth, direction, values.size(), index);
          assert(twiddleBits && "verified CFFT extent must have a static twiddle table");
          twiddle = createPackedTwiddle(*twiddleBits);
        }
        auto plain = createButterfly(even[index], odd[index], twiddle,
                                     exact ? ondrix::ondsp::CxButterflyVariant::Unit
                                           : ondrix::ondsp::CxButterflyVariant::Plain);
        outputs[index] = plain.getOut0();
        outputs[index + half] = plain.getOut1();
        auto crossLeg = createButterfly(even[index + half / 2], odd[index + half / 2], twiddle,
                                        exact ? ondrix::ondsp::CxButterflyVariant::UnitCross
                                              : ondrix::ondsp::CxButterflyVariant::Cross);
        outputs[index + half / 2] = crossLeg.getOut0();
        outputs[index + half / 2 + half] = crossLeg.getOut1();
      }
      return outputs;
    }
    if (vectorizeStaticCfft && !inventoryPaired && even.size() > 1) {
      SmallVector<Value> twiddles;
      twiddles.reserve(even.size());
      for (int64_t index = 0, end = even.size(); index < end; ++index) {
        std::optional<uint64_t> twiddleBits =
            getPackedTwiddleBits(profile.storageWidth, direction, values.size(), index);
        assert(twiddleBits && "verified CFFT extent must have a static twiddle table");
        twiddles.push_back(createPackedTwiddle(*twiddleBits));
      }
      Value evenVector = buildVector(even);
      Value oddVector = buildVector(odd);
      Value twiddleVector = buildVector(twiddles);
      auto vectorType = cast<VectorType>(evenVector.getType());
      auto butterfly = rewriter.create<ondrix::ondsp::CxButterflyOp>(
          loc, vectorType, vectorType, evenVector, oddVector, twiddleVector, layout, numeric,
          product, productScale, outputScale, ondrix::ondsp::CxButterflyVariantAttr());
      for (int64_t index = 0, end = even.size(); index < end; ++index) {
        outputs[index] = rewriter.create<vector::ExtractOp>(loc, butterfly.getOut0(), index);
        outputs[index + end] = rewriter.create<vector::ExtractOp>(loc, butterfly.getOut1(), index);
      }
      return outputs;
    }
    for (int64_t index = 0, end = values.size() / 2; index < end; ++index) {
      std::optional<uint64_t> twiddleBits =
          getPackedTwiddleBits(profile.storageWidth, direction, values.size(), index);
      assert(twiddleBits && "verified CFFT extent must have a static twiddle table");
      auto butterfly = createButterfly(even[index], odd[index], createPackedTwiddle(*twiddleBits));
      outputs[index] = butterfly.getOut0();
      outputs[index + end] = butterfly.getOut1();
    }
    return outputs;
  };
  return lowerCfft(inputs);
}

// Loop-form lowering of the same recursive DIT dataflow with in-memory
// tables. The recursive even/odd combine applied to natural-order input
// computes exactly the butterflies of the iterative algorithm applied to
// bit-reversed input: stage half-sizes H = 1, 2, ..., N/2, butterfly j of
// group g pairing positions g*2H + j and g*2H + j + H under twiddle
// W(2H, j). Every butterfly is the same scalar ondsp.cx_butterfly with the
// same requantization attributes, so each output element passes through the
// identical sequence of quantization boundaries in an equivalent order and
// the two lowerings are bit-identical per element; only the code shape
// changes (loops and constant tables instead of unrolled SSA butterflies).
// Every width follows from the profile: the packed container carries both the
// values and the twiddle table, and only the 16-bit packed target has the
// paired inventory form.
static Value
lowerPackedCfftLoops(Location loc, Value input, int64_t extent, ondrix::ir::CfftDirection direction,
                     ondrix::ondsp::PackedComplexProfile profile,
                     ondrix::ondsp::CxLayoutAttr layout, Attribute numeric,
                     ondrix::ondsp::ProductAttr product, ondrix::ondsp::ScaleAttr productScale,
                     ondrix::ondsp::ScaleAttr outputScale, ConversionPatternRewriter &rewriter) {
  IntegerType container = rewriter.getIntegerType(profile.containerWidth);
  IntegerType i64 = rewriter.getI64Type();
  int64_t stageCount = llvm::Log2_64(extent);
  bool inventoryPaired = profile.storageWidth == 16 &&
                         isInventoryRounding(productScale.getRounding()) &&
                         isInventoryRounding(outputScale.getRounding());
  // The exact unit twiddle of the profile, the value the unit variants ignore.
  auto unitTwiddle = [&](OpBuilder &builder) {
    return builder.create<arith::ConstantOp>(
        loc, builder.getIntegerAttr(container, (int64_t(1) << (profile.storageWidth - 1)) - 1));
  };

  // Generic form: twiddles[H + j] = W(2H, j) for j < H. Paired inventory
  // form: twiddles[H/2 + j] = W(2H, j) for j < H/2, H >= 4 - only the first
  // quarter turn is stored, the second leg of each pair reuses it through
  // the cross combine, -j never appears, and the exact unit stages H = 1, 2
  // read no table at all (slots 0 and 1 stay zero).
  SmallVector<APInt> twiddleWords(inventoryPaired ? extent / 2 : extent,
                                  APInt::getZero(profile.containerWidth));
  for (int64_t half = inventoryPaired ? 4 : 1; half < extent; half *= 2) {
    int64_t count = inventoryPaired ? half / 2 : half;
    for (int64_t index = 0; index < count; ++index) {
      std::optional<uint64_t> bits =
          getPackedTwiddleBits(profile.storageWidth, direction, 2 * half, index);
      assert(bits && "twiddle admissibility was checked before lowering");
      twiddleWords[(inventoryPaired ? half / 2 : half) + index] =
          APInt(profile.containerWidth, *bits);
    }
  }
  Value twiddleTable = rewriter.create<arith::ConstantOp>(
      loc, DenseElementsAttr::get(
               RankedTensorType::get({static_cast<int64_t>(twiddleWords.size())}, container),
               twiddleWords));

  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value extentValue = rewriter.create<arith::ConstantIndexOp>(loc, extent);
  Value halfExtent = rewriter.create<arith::ConstantIndexOp>(loc, extent / 2);
  Value stages = rewriter.create<arith::ConstantIndexOp>(loc, stageCount);

  Value empty = rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{extent}, container);
  // The paired form needs no separate permute loop or reversal table: the
  // fused group loop below reads the input through a loop-carried
  // reversed-carry cursor. The generic form keeps the table gather as its
  // own loop.
  Value permuted = empty;
  if (!inventoryPaired) {
    SmallVector<int64_t> bitReversed(extent);
    for (int64_t index = 0; index < extent; ++index) {
      int64_t reversed = 0;
      for (int64_t bit = 0; bit < stageCount; ++bit)
        reversed |= ((index >> bit) & 1) << (stageCount - 1 - bit);
      bitReversed[index] = reversed;
    }
    Value reversalTable = rewriter.create<arith::ConstantOp>(
        loc, DenseElementsAttr::get(RankedTensorType::get({extent}, i64),
                                    llvm::ArrayRef<int64_t>(bitReversed)));
    auto permuteLoop = rewriter.create<scf::ForOp>(
        loc, zero, extentValue, one, ValueRange{empty},
        [&](OpBuilder &builder, Location loc, Value position, ValueRange iterArgs) {
          Value source64 = builder.create<tensor::ExtractOp>(loc, reversalTable, position);
          Value source = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), source64);
          Value value = builder.create<tensor::ExtractOp>(loc, input, source);
          Value inserted = builder.create<tensor::InsertOp>(loc, value, iterArgs.front(), position);
          builder.create<scf::YieldOp>(loc, inserted);
        });
    permuted = permuteLoop.getResult(0);
  }

  // One leg: extract the pair at (upper, upper + half), butterfly it with
  // twiddles[twiddleIndex], and insert both outputs back. The exact unit
  // variants read no table; their ignored twiddle operand is the unit word.
  auto buildLeg = [&](OpBuilder &builder, Location loc, Value data, Value half, Value upper,
                      Value twiddleIndex, ondrix::ondsp::CxButterflyVariant variant) -> Value {
    Value lower = builder.create<arith::AddIOp>(loc, upper, half);
    Value a = builder.create<tensor::ExtractOp>(loc, data, upper);
    Value b = builder.create<tensor::ExtractOp>(loc, data, lower);
    bool unit = variant == ondrix::ondsp::CxButterflyVariant::Unit ||
                variant == ondrix::ondsp::CxButterflyVariant::UnitCross;
    Value twiddle =
        unit ? unitTwiddle(builder).getResult()
             : builder.create<tensor::ExtractOp>(loc, twiddleTable, twiddleIndex).getResult();
    auto attr = variant == ondrix::ondsp::CxButterflyVariant::Plain
                    ? ondrix::ondsp::CxButterflyVariantAttr()
                    : ondrix::ondsp::CxButterflyVariantAttr::get(builder.getContext(), variant);
    auto butterfly = builder.create<ondrix::ondsp::CxButterflyOp>(loc, container, container, a, b,
                                                                  twiddle, layout, numeric, product,
                                                                  productScale, outputScale, attr);
    Value insertUpper = builder.create<tensor::InsertOp>(loc, butterfly.getOut0(), data, upper);
    return builder.create<tensor::InsertOp>(loc, butterfly.getOut1(), insertUpper, lower);
  };

  // The exact unit stages H = 1 and H = 2 fuse into one group loop ahead
  // of the dynamic stage loop (decision 2026-08-17): each group of four
  // elements passes both exact stages in registers, so the body reads no
  // twiddle table, carries only fixed variants, and stays branch-free.
  Value current = permuted;
  Value lowerStage = zero;
  if (inventoryPaired) {
    Value two = rewriter.create<arith::ConstantIndexOp>(loc, 2);
    Value three = rewriter.create<arith::ConstantIndexOp>(loc, 3);
    Value four = rewriter.create<arith::ConstantIndexOp>(loc, 4);
    Value quarter = rewriter.create<arith::ConstantIndexOp>(loc, extent / 4);
    auto groupLoop = rewriter.create<scf::ForOp>(
        loc, zero, quarter, one, ValueRange{current, zero},
        [&](OpBuilder &builder, Location loc, Value group, ValueRange groupArgs) {
          Value base = builder.create<arith::MulIOp>(loc, group, four);
          Value at1 = builder.create<arith::AddIOp>(loc, base, one);
          Value at2 = builder.create<arith::AddIOp>(loc, base, two);
          Value at3 = builder.create<arith::AddIOp>(loc, base, three);
          Value data = groupArgs.front();
          // The carried cursor walks rev(0), rev(1), ... across the whole
          // loop, so group g reads input[rev(4g + i)] - the same values the
          // dropped reversal-table gather produced.
          Value cursor = groupArgs.back();
          auto walkLoad = [&]() {
            Value value = builder.create<tensor::ExtractOp>(loc, input, cursor);
            cursor = builder.create<ondrix::ondsp::BitrevAddOp>(loc, builder.getIndexType(), cursor,
                                                                halfExtent, uint64_t(stageCount));
            return value;
          };
          Value x0 = walkLoad();
          Value x1 = walkLoad();
          Value x2 = walkLoad();
          Value x3 = walkLoad();
          Value unitWord = unitTwiddle(builder);
          auto unitButterfly = [&](Value a, Value b, ondrix::ondsp::CxButterflyVariant variant) {
            return builder.create<ondrix::ondsp::CxButterflyOp>(
                loc, container, container, a, b, unitWord, layout, numeric, product, productScale,
                outputScale,
                ondrix::ondsp::CxButterflyVariantAttr::get(builder.getContext(), variant));
          };
          auto pairA = unitButterfly(x0, x1, ondrix::ondsp::CxButterflyVariant::Unit);
          auto pairB = unitButterfly(x2, x3, ondrix::ondsp::CxButterflyVariant::Unit);
          auto plusLeg = unitButterfly(pairA.getOut0(), pairB.getOut0(),
                                       ondrix::ondsp::CxButterflyVariant::Unit);
          auto crossLeg = unitButterfly(pairA.getOut1(), pairB.getOut1(),
                                        ondrix::ondsp::CxButterflyVariant::UnitCross);
          Value out = builder.create<tensor::InsertOp>(loc, plusLeg.getOut0(), data, base);
          out = builder.create<tensor::InsertOp>(loc, crossLeg.getOut0(), out, at1);
          out = builder.create<tensor::InsertOp>(loc, plusLeg.getOut1(), out, at2);
          out = builder.create<tensor::InsertOp>(loc, crossLeg.getOut1(), out, at3);
          builder.create<scf::YieldOp>(loc, ValueRange{out, cursor});
        });
    current = groupLoop.getResult(0);
    lowerStage = two;
  }

  auto stageLoop = rewriter.create<scf::ForOp>(
      loc, lowerStage, stages, one, ValueRange{current},
      [&](OpBuilder &builder, Location loc, Value stage, ValueRange stageArgs) {
        Value half = builder.create<arith::ShLIOp>(loc, one, stage);
        Value doubled = builder.create<arith::AddIOp>(loc, half, half);
        if (!inventoryPaired) {
          auto butterflyLoop = builder.create<scf::ForOp>(
              loc, zero, halfExtent, one, ValueRange{stageArgs.front()},
              [&](OpBuilder &builder, Location loc, Value pair, ValueRange pairArgs) {
                Value group = builder.create<arith::DivUIOp>(loc, pair, half);
                Value phase = builder.create<arith::RemUIOp>(loc, pair, half);
                Value base = builder.create<arith::MulIOp>(loc, group, doubled);
                Value upper = builder.create<arith::AddIOp>(loc, base, phase);
                Value twiddleIndex = builder.create<arith::AddIOp>(loc, half, phase);
                builder.create<scf::YieldOp>(
                    loc, buildLeg(builder, loc, pairArgs.front(), half, upper, twiddleIndex,
                                  ondrix::ondsp::CxButterflyVariant::Plain));
              });
          builder.create<scf::YieldOp>(loc, butterflyLoop.getResult(0));
          return;
        }
        // Paired form as group-nested unit-stride loops: each inner body
        // carries one fixed variant and walks its data and twiddle streams
        // at stride one, and the group-major order matches the flat legs
        // it replaces element for element. Both legs of a pair read
        // twiddles[H/2 + j]; only H >= 4 reaches here.
        Value halfHalf = builder.create<arith::ShRUIOp>(loc, half, one);
        Value groups = builder.create<arith::DivUIOp>(loc, extentValue, doubled);
        auto legLoops = [&](Value data, Value phaseBase,
                            ondrix::ondsp::CxButterflyVariant variant) -> Value {
          auto groupLoop = builder.create<scf::ForOp>(
              loc, zero, groups, one, ValueRange{data},
              [&](OpBuilder &builder, Location loc, Value group, ValueRange groupArgs) {
                Value base = builder.create<arith::MulIOp>(loc, group, doubled);
                Value start = builder.create<arith::AddIOp>(loc, base, phaseBase);
                auto innerLoop = builder.create<scf::ForOp>(
                    loc, zero, halfHalf, one, ValueRange{groupArgs.front()},
                    [&](OpBuilder &builder, Location loc, Value j, ValueRange innerArgs) {
                      Value upper = builder.create<arith::AddIOp>(loc, start, j);
                      Value twiddleIndex = builder.create<arith::AddIOp>(loc, halfHalf, j);
                      builder.create<scf::YieldOp>(loc,
                                                   buildLeg(builder, loc, innerArgs.front(), half,
                                                            upper, twiddleIndex, variant));
                    });
                builder.create<scf::YieldOp>(loc, innerLoop.getResult(0));
              });
          return groupLoop.getResult(0);
        };
        Value afterPlain =
            legLoops(stageArgs.front(), zero, ondrix::ondsp::CxButterflyVariant::Plain);
        Value afterCross = legLoops(afterPlain, halfHalf, ondrix::ondsp::CxButterflyVariant::Cross);
        builder.create<scf::YieldOp>(loc, afterCross);
      });
  return stageLoop.getResult(0);
}

static Value canonicalizePackedReal(Location loc, Value packed,
                                    ondrix::ondsp::PackedComplexProfile profile,
                                    OpBuilder &rewriter) {
  IntegerType storage = rewriter.getIntegerType(profile.storageWidth);
  IntegerType container = rewriter.getIntegerType(profile.containerWidth);
  Value real = rewriter.create<arith::TruncIOp>(loc, storage, packed);
  return rewriter.create<arith::ExtUIOp>(loc, container, real);
}

static Value conjugatePackedSaturating(Location loc, Value packed,
                                       ondrix::ondsp::PackedComplexProfile profile,
                                       OpBuilder &rewriter) {
  IntegerType storage = rewriter.getIntegerType(profile.storageWidth);
  IntegerType container = rewriter.getIntegerType(profile.containerWidth);
  int64_t storageMinimum = -(int64_t(1) << (profile.storageWidth - 1));
  Value real = rewriter.create<arith::TruncIOp>(loc, storage, packed);
  Value shift =
      rewriter.create<arith::ConstantIntOp>(loc, profile.storageWidth, profile.containerWidth);
  Value high = rewriter.create<arith::ShRUIOp>(loc, packed, shift);
  Value imaginary = rewriter.create<arith::TruncIOp>(loc, storage, high);
  Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, profile.storageWidth);
  Value minimum = rewriter.create<arith::ConstantIntOp>(loc, storageMinimum, profile.storageWidth);
  Value maximum =
      rewriter.create<arith::ConstantIntOp>(loc, -(storageMinimum + 1), profile.storageWidth);
  Value isMinimum =
      rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, imaginary, minimum);
  Value negated = rewriter.create<arith::SubIOp>(loc, zero, imaginary);
  Value conjugatedImaginary = rewriter.create<arith::SelectOp>(loc, isMinimum, maximum, negated);
  Value realBits = rewriter.create<arith::ExtUIOp>(loc, container, real);
  Value imaginaryBits = rewriter.create<arith::ExtUIOp>(loc, container, conjugatedImaginary);
  Value shiftedImaginary = rewriter.create<arith::ShLIOp>(loc, imaginaryBits, shift);
  return rewriter.create<arith::OrIOp>(loc, shiftedImaginary, realBits);
}

class CfftOpLowering final : public OpConversionPattern<ondrix::ir::CfftOp> {
public:
  CfftOpLowering(MLIRContext *context, bool vectorizeStaticCfft, bool fftLoops)
      : OpConversionPattern(context), vectorizeStaticCfft(vectorizeStaticCfft), fftLoops(fftLoops) {
  }

  LogicalResult matchAndRewrite(ondrix::ir::CfftOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto layout = dyn_cast<ondrix::ondsp::CxLayoutAttr>(op.getLayout());
    if (!layout)
      return rewriter.notifyMatchFailure(op, "requires an ondsp.cx_layout layout attribute");

    std::optional<ondrix::ondsp::PackedComplexProfile> profile =
        ondrix::ondsp::getPackedComplexProfile(layout.getLayout());
    if (!profile)
      return rewriter.notifyMatchFailure(op, "layout has no executable packed complex profile");
    int64_t extent = op.getInput().getType().getDimSize(0);
    if (!hasAdmissiblePackedTwiddleTables(profile->storageWidth, op.getDirection(), extent))
      return rewriter.notifyMatchFailure(op, "the stage twiddle table is unavailable");
    if (fftLoops) {
      Value result = lowerPackedCfftLoops(loc, adaptor.getInput(), extent, op.getDirection(),
                                          *profile, layout, op.getNumeric(), op.getProduct(),
                                          op.getProductScale(), op.getOutputScale(), rewriter);
      rewriter.replaceOp(op, result);
      return success();
    }
    SmallVector<Value> indices;
    SmallVector<Value> inputs;
    indices.reserve(extent);
    inputs.reserve(extent);
    for (int64_t index = 0; index < extent; ++index) {
      Value position = rewriter.create<arith::ConstantIndexOp>(loc, index);
      indices.push_back(position);
      inputs.push_back(rewriter.create<tensor::ExtractOp>(loc, adaptor.getInput(), position));
    }
    SmallVector<Value> outputs = lowerPackedCfft(
        loc, inputs, op.getDirection(), *profile, layout, op.getNumeric(), op.getProduct(),
        op.getProductScale(), op.getOutputScale(), vectorizeStaticCfft, rewriter);

    Value result = rewriter.create<tensor::EmptyOp>(loc, op.getResult().getType().getShape(),
                                                    op.getResult().getType().getElementType());
    for (auto [value, position] : llvm::zip_equal(outputs, indices))
      result = rewriter.create<tensor::InsertOp>(loc, value, result, position);
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  bool vectorizeStaticCfft;
  bool fftLoops;
};

class RfftOpLowering final : public OpConversionPattern<ondrix::ir::RfftOp> {
public:
  RfftOpLowering(MLIRContext *context, bool vectorizeStaticCfft, bool fftLoops)
      : OpConversionPattern(context), vectorizeStaticCfft(vectorizeStaticCfft), fftLoops(fftLoops) {
  }

  LogicalResult matchAndRewrite(ondrix::ir::RfftOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ondrix::ondsp::PackedComplexProfile profile = getVerifiedPackedProfile(op.getLayout());
    IntegerType container = rewriter.getIntegerType(profile.containerWidth);
    int64_t extent = op.getInput().getType().getDimSize(0);
    if (!hasAdmissiblePackedTwiddleTables(profile.storageWidth, ondrix::ir::CfftDirection::Forward,
                                          extent))
      return rewriter.notifyMatchFailure(op, "the stage twiddle table is unavailable");
    if (fftLoops) {
      Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      Value extentValue = rewriter.create<arith::ConstantIndexOp>(loc, extent);
      Value empty = rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{extent}, container);
      auto packLoop = rewriter.create<scf::ForOp>(
          loc, zero, extentValue, one, ValueRange{empty},
          [&](OpBuilder &builder, Location loc, Value position, ValueRange iterArgs) {
            Value real = builder.create<tensor::ExtractOp>(loc, adaptor.getInput(), position);
            Value packed = builder.create<arith::ExtUIOp>(loc, container, real);
            Value inserted =
                builder.create<tensor::InsertOp>(loc, packed, iterArgs.front(), position);
            builder.create<scf::YieldOp>(loc, inserted);
          });
      Value spectrum = lowerPackedCfftLoops(loc, packLoop.getResult(0), extent,
                                            ondrix::ir::CfftDirection::Forward, profile,
                                            op.getLayout(), op.getNumeric(), op.getProduct(),
                                            op.getProductScale(), op.getOutputScale(), rewriter);
      RankedTensorType resultType = op.getResult().getType();
      int64_t binCount = resultType.getDimSize(0);
      Value compact = rewriter.create<tensor::ExtractSliceOp>(
          loc, resultType, spectrum, ArrayRef<OpFoldResult>{rewriter.getIndexAttr(0)},
          ArrayRef<OpFoldResult>{rewriter.getIndexAttr(binCount)},
          ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)});
      Value half = rewriter.create<arith::ConstantIndexOp>(loc, extent / 2);
      Value dc = rewriter.create<tensor::ExtractOp>(loc, compact, zero);
      compact = rewriter.create<tensor::InsertOp>(
          loc, canonicalizePackedReal(loc, dc, profile, rewriter), compact, zero);
      Value nyquist = rewriter.create<tensor::ExtractOp>(loc, compact, half);
      compact = rewriter.create<tensor::InsertOp>(
          loc, canonicalizePackedReal(loc, nyquist, profile, rewriter), compact, half);
      rewriter.replaceOp(op, compact);
      return success();
    }
    SmallVector<Value> inputs;
    inputs.reserve(extent);
    for (int64_t index = 0; index < extent; ++index) {
      Value position = rewriter.create<arith::ConstantIndexOp>(loc, index);
      Value real = rewriter.create<tensor::ExtractOp>(loc, adaptor.getInput(), position);
      inputs.push_back(rewriter.create<arith::ExtUIOp>(loc, container, real));
    }

    SmallVector<Value> outputs = lowerPackedCfft(
        loc, inputs, ondrix::ir::CfftDirection::Forward, profile, op.getLayout(), op.getNumeric(),
        op.getProduct(), op.getProductScale(), op.getOutputScale(), vectorizeStaticCfft, rewriter);
    outputs.front() = canonicalizePackedReal(loc, outputs.front(), profile, rewriter);
    outputs[extent / 2] = canonicalizePackedReal(loc, outputs[extent / 2], profile, rewriter);

    RankedTensorType resultType = op.getResult().getType();
    Value result =
        rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
    for (int64_t index = 0, end = resultType.getDimSize(0); index < end; ++index) {
      Value position = rewriter.create<arith::ConstantIndexOp>(loc, index);
      result = rewriter.create<tensor::InsertOp>(loc, outputs[index], result, position);
    }
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  bool vectorizeStaticCfft;
  bool fftLoops;
};

class IrfftOpLowering final : public OpConversionPattern<ondrix::ir::IrfftOp> {
public:
  IrfftOpLowering(MLIRContext *context, bool vectorizeStaticCfft, bool fftLoops)
      : OpConversionPattern(context), vectorizeStaticCfft(vectorizeStaticCfft), fftLoops(fftLoops) {
  }

  LogicalResult matchAndRewrite(ondrix::ir::IrfftOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ondrix::ondsp::PackedComplexProfile profile = getVerifiedPackedProfile(op.getLayout());
    IntegerType storage = rewriter.getIntegerType(profile.storageWidth);
    int64_t extent = op.getResult().getType().getDimSize(0);
    if (!hasAdmissiblePackedTwiddleTables(profile.storageWidth, ondrix::ir::CfftDirection::Inverse,
                                          extent))
      return rewriter.notifyMatchFailure(op, "the stage twiddle table is unavailable");
    int64_t half = extent / 2;
    if (fftLoops) {
      IntegerType container = rewriter.getIntegerType(profile.containerWidth);
      Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      Value extentValue = rewriter.create<arith::ConstantIndexOp>(loc, extent);
      Value halfValue = rewriter.create<arith::ConstantIndexOp>(loc, half);
      Value empty = rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{extent}, container);
      Value dc = rewriter.create<tensor::ExtractOp>(loc, adaptor.getInput(), zero);
      Value seeded = rewriter.create<tensor::InsertOp>(
          loc, canonicalizePackedReal(loc, dc, profile, rewriter), empty, zero);
      Value nyquist = rewriter.create<tensor::ExtractOp>(loc, adaptor.getInput(), halfValue);
      seeded = rewriter.create<tensor::InsertOp>(
          loc, canonicalizePackedReal(loc, nyquist, profile, rewriter), seeded, halfValue);
      auto mirrorLoop = rewriter.create<scf::ForOp>(
          loc, one, halfValue, one, ValueRange{seeded},
          [&](OpBuilder &builder, Location loc, Value position, ValueRange iterArgs) {
            Value bin = builder.create<tensor::ExtractOp>(loc, adaptor.getInput(), position);
            Value direct = builder.create<tensor::InsertOp>(loc, bin, iterArgs.front(), position);
            Value mirrored = builder.create<arith::SubIOp>(loc, extentValue, position);
            Value conjugated = conjugatePackedSaturating(loc, bin, profile, builder);
            Value full = builder.create<tensor::InsertOp>(loc, conjugated, direct, mirrored);
            builder.create<scf::YieldOp>(loc, full);
          });
      Value outputs = lowerPackedCfftLoops(loc, mirrorLoop.getResult(0), extent,
                                           ondrix::ir::CfftDirection::Inverse, profile,
                                           op.getLayout(), op.getNumeric(), op.getProduct(),
                                           op.getProductScale(), op.getOutputScale(), rewriter);
      RankedTensorType resultType = op.getResult().getType();
      Value resultEmpty =
          rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
      auto truncateLoop = rewriter.create<scf::ForOp>(
          loc, zero, extentValue, one, ValueRange{resultEmpty},
          [&](OpBuilder &builder, Location loc, Value position, ValueRange iterArgs) {
            Value packed = builder.create<tensor::ExtractOp>(loc, outputs, position);
            Value real = builder.create<arith::TruncIOp>(loc, storage, packed);
            Value inserted =
                builder.create<tensor::InsertOp>(loc, real, iterArgs.front(), position);
            builder.create<scf::YieldOp>(loc, inserted);
          });
      rewriter.replaceOp(op, truncateLoop.getResult(0));
      return success();
    }
    SmallVector<Value> compact;
    compact.reserve(half + 1);
    for (int64_t index = 0; index <= half; ++index) {
      Value position = rewriter.create<arith::ConstantIndexOp>(loc, index);
      compact.push_back(rewriter.create<tensor::ExtractOp>(loc, adaptor.getInput(), position));
    }

    SmallVector<Value> spectrum(extent);
    spectrum.front() = canonicalizePackedReal(loc, compact.front(), profile, rewriter);
    spectrum[half] = canonicalizePackedReal(loc, compact[half], profile, rewriter);
    for (int64_t index = 1; index < half; ++index) {
      spectrum[index] = compact[index];
      spectrum[extent - index] = conjugatePackedSaturating(loc, compact[index], profile, rewriter);
    }

    SmallVector<Value> outputs = lowerPackedCfft(
        loc, spectrum, ondrix::ir::CfftDirection::Inverse, profile, op.getLayout(), op.getNumeric(),
        op.getProduct(), op.getProductScale(), op.getOutputScale(), vectorizeStaticCfft, rewriter);
    RankedTensorType resultType = op.getResult().getType();
    Value result =
        rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
    for (int64_t index = 0; index < extent; ++index) {
      Value position = rewriter.create<arith::ConstantIndexOp>(loc, index);
      Value real = rewriter.create<arith::TruncIOp>(loc, storage, outputs[index]);
      result = rewriter.create<tensor::InsertOp>(loc, real, result, position);
    }
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  bool vectorizeStaticCfft;
  bool fftLoops;
};

// One complex value of a half-size split schedule, carried as two i32 SSA
// components between the explicit ondsp requantization points.
struct SplitComplexValue {
  Value real;
  Value imaginary;
};

class RfftRadix4SplitOpLowering final : public OpConversionPattern<ondrix::ir::RfftRadix4SplitOp> {
public:
  using OpConversionPattern<ondrix::ir::RfftRadix4SplitOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ir::RfftRadix4SplitOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();
    IntegerType i16 = rewriter.getI16Type();
    IntegerType i32 = rewriter.getI32Type();

    auto constant = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantIntOp>(loc, value, 32);
    };
    // Every rounding decision of the schedule is an explicit toward-negative
    // ondsp.round_shift; the i32 carrier width never narrows here, so the
    // scale's overflow mode is unreachable.
    auto floorShift = [&](Value value, unsigned shift) -> Value {
      auto scale = ondrix::ondsp::ScaleAttr::get(context, 0, shift,
                                                 ondrix::ondsp::RoundingMode::TowardNegative,
                                                 ondrix::ondsp::OverflowMode::Wrap, i32);
      return rewriter.create<ondrix::ondsp::RoundShiftOp>(loc, i32, value, scale);
    };
    // The schedule's only reachable clamps: the stage-two saturating
    // combines on 4.12-format values, widened back for exact arithmetic.
    auto saturate16 = [&](Value value) -> Value {
      auto numeric =
          ondrix::ondsp::FixedAttr::get(context, ondrix::ondsp::Signedness::Signed, i16, 12);
      Value clamped = rewriter.create<ondrix::ondsp::SatCastOp>(loc, i16, value, numeric);
      return rewriter.create<arith::ExtSIOp>(loc, i32, clamped);
    };
    auto add = [&](Value lhs, Value rhs) -> Value {
      return rewriter.create<arith::AddIOp>(loc, lhs, rhs);
    };
    auto sub = [&](Value lhs, Value rhs) -> Value {
      return rewriter.create<arith::SubIOp>(loc, lhs, rhs);
    };
    auto mul = [&](int64_t coefficient, Value value) -> Value {
      return rewriter.create<arith::MulIOp>(loc, constant(coefficient), value);
    };
    // out = (u + j*v) * (co - j*si): exact i32 cross products, then one
    // sixteen-bit floor shift. The i16 narrowing recorded in the contract is
    // proven exact (P2/P4), so the value legitimately stays in its carrier.
    auto twiddle = [&](Value u, Value v, int64_t co, int64_t si) -> SplitComplexValue {
      Value real = floorShift(add(mul(co, u), mul(si, v)), 16);
      Value imaginary = floorShift(sub(mul(co, v), mul(si, u)), 16);
      return {real, imaginary};
    };

    // Frozen Q15 twiddle pairs (pair index -> co, si); pairs 5, 7, and 8 are
    // never consumed at this length and stay zero placeholders.
    static constexpr int64_t kTwiddles[10][2] = {
        {32767, 0}, {30273, 12539},  {23170, 23170}, {12539, 30273}, {0, 32767},
        {0, 0},     {-23171, 23170}, {0, 0},         {0, 0},         {-30274, -12540}};
    // Frozen split coefficients (bin -> Ar, Ai, Br, Bi); bin 0 is unused.
    static constexpr int64_t kSplitCoefficients[16][4] = {{0, 0, 0, 0},
                                                          {13188, -16069, 19580, 16069},
                                                          {10114, -15137, 22654, 15137},
                                                          {7282, -13623, 25486, 13623},
                                                          {4799, -11585, 27969, 11585},
                                                          {2761, -9102, 30007, 9102},
                                                          {1247, -6270, 31521, 6270},
                                                          {315, -3196, 32453, 3196},
                                                          {0, 0, 32767, 0},
                                                          {315, 3196, 32453, -3196},
                                                          {1247, 6270, 31521, -6270},
                                                          {2761, 9102, 30007, -9102},
                                                          {4799, 11585, 27969, -11585},
                                                          {7282, 13623, 25486, -13623},
                                                          {10114, 15137, 22654, -15137},
                                                          {13188, 16069, 19580, -16069}};

    // View the 32 real Q1.15 samples as 16 complex values.
    SmallVector<Value> samples;
    samples.reserve(32);
    for (int64_t index = 0; index < 32; ++index) {
      Value position = rewriter.create<arith::ConstantIndexOp>(loc, index);
      Value element = rewriter.create<tensor::ExtractOp>(loc, adaptor.getInput(), position);
      samples.push_back(rewriter.create<arith::ExtSIOp>(loc, i32, element));
    }
    SmallVector<SplitComplexValue> values(16);
    for (int64_t m = 0; m < 16; ++m)
      values[m] = {samples[2 * m], samples[2 * m + 1]};

    // Stage 1: radix-4 groups (g, g+4, g+8, g+12), twiddle pair index g.
    // The fourteen per-group clamps recorded in the contract are proven
    // inactive (proof P1), so they are omitted without changing any bit.
    for (int64_t group = 0; group < 4; ++group) {
      int64_t a = group, b = group + 4, c = group + 8, d = group + 12;
      Value t0 = floorShift(values[a].real, 2);
      Value t1 = floorShift(values[a].imaginary, 2);
      Value s0 = floorShift(values[c].real, 2);
      Value s1 = floorShift(values[c].imaginary, 2);
      Value sum0 = add(t0, s0);
      Value sum1 = add(t1, s1);
      Value diff0 = sub(t0, s0);
      Value diff1 = sub(t1, s1);
      Value tb0 = floorShift(values[b].real, 2);
      Value tb1 = floorShift(values[b].imaginary, 2);
      Value u0 = floorShift(values[d].real, 2);
      Value u1 = floorShift(values[d].imaginary, 2);
      Value tSum0 = add(tb0, u0);
      Value tSum1 = add(tb1, u1);
      values[a] = {add(floorShift(sum0, 1), floorShift(tSum0, 1)),
                   add(floorShift(sum1, 1), floorShift(tSum1, 1))};
      Value r0 = sub(sum0, tSum0);
      Value r1 = sub(sum1, tSum1);
      values[b] = twiddle(r0, r1, kTwiddles[2 * group][0], kTwiddles[2 * group][1]);
      Value tDiff0 = sub(tb0, u0);
      Value tDiff1 = sub(tb1, u1);
      Value rr0 = sub(diff0, tDiff1);
      Value rr1 = add(diff1, tDiff0);
      Value ss0 = add(diff0, tDiff1);
      Value ss1 = sub(diff1, tDiff0);
      values[c] = twiddle(ss0, ss1, kTwiddles[group][0], kTwiddles[group][1]);
      values[d] = twiddle(rr0, rr1, kTwiddles[3 * group][0], kTwiddles[3 * group][1]);
    }

    // Stage 2: unit-twiddle radix-4 groups (i, i+1, i+2, i+3). Saturating
    // combine first, then independent one-bit floor shifts; these are the
    // schedule's only reachable saturation points.
    for (int64_t group = 0; group < 16; group += 4) {
      SplitComplexValue za = values[group], zb = values[group + 1], zc = values[group + 2],
                        zd = values[group + 3];
      Value r0 = saturate16(add(za.real, zc.real));
      Value r1 = saturate16(add(za.imaginary, zc.imaginary));
      Value s0 = saturate16(sub(za.real, zc.real));
      Value s1 = saturate16(sub(za.imaginary, zc.imaginary));
      Value tSum0 = saturate16(add(zb.real, zd.real));
      Value tSum1 = saturate16(add(zb.imaginary, zd.imaginary));
      Value halfR0 = floorShift(r0, 1);
      Value halfR1 = floorShift(r1, 1);
      Value halfT0 = floorShift(tSum0, 1);
      Value halfT1 = floorShift(tSum1, 1);
      values[group] = {add(halfR0, halfT0), add(halfR1, halfT1)};
      values[group + 1] = {sub(halfR0, halfT0), sub(halfR1, halfT1)};
      Value tDiff0 = saturate16(sub(zb.real, zd.real));
      Value tDiff1 = saturate16(sub(zb.imaginary, zd.imaginary));
      Value halfS0 = floorShift(s0, 1);
      Value halfS1 = floorShift(s1, 1);
      Value halfD0 = floorShift(tDiff0, 1);
      Value halfD1 = floorShift(tDiff1, 1);
      values[group + 2] = {add(halfS0, halfD1), sub(halfS1, halfD0)};
      values[group + 3] = {sub(halfS0, halfD1), add(halfS1, halfD0)};
    }

    // Binary bit reversal: swap the six non-fixed orbits.
    static constexpr int64_t kBitReversalPairs[6][2] = {{1, 8},  {2, 4},  {3, 12},
                                                        {5, 10}, {7, 14}, {11, 13}};
    for (const auto &pair : kBitReversalPairs)
      std::swap(values[pair[0]], values[pair[1]]);

    auto packBin = [&](Value real, Value imaginary) -> Value {
      Value realBits = rewriter.create<arith::ExtUIOp>(
          loc, i32, rewriter.create<arith::TruncIOp>(loc, i16, real));
      Value imaginaryBits = rewriter.create<arith::ExtUIOp>(
          loc, i32, rewriter.create<arith::TruncIOp>(loc, i16, imaginary));
      Value shifted = rewriter.create<arith::ShLIOp>(loc, imaginaryBits, constant(16));
      return rewriter.create<arith::OrIOp>(loc, shifted, realBits);
    };

    // Split stage into compact natural-order bins 0..16; DC and Nyquist
    // imaginary components are exactly zero by construction.
    SmallVector<Value> bins(17);
    bins[0] = packBin(floorShift(add(values[0].real, values[0].imaginary), 1), constant(0));
    bins[16] = packBin(floorShift(sub(values[0].real, values[0].imaginary), 1), constant(0));
    for (int64_t k = 1; k < 16; ++k) {
      const auto &coefficient = kSplitCoefficients[k];
      SplitComplexValue z = values[k];
      SplitComplexValue w = values[16 - k];
      Value accR = add(add(sub(mul(coefficient[0], z.real), mul(coefficient[1], z.imaginary)),
                           mul(coefficient[2], w.real)),
                       mul(coefficient[3], w.imaginary));
      Value accI = add(add(sub(mul(coefficient[3], w.real), mul(coefficient[2], w.imaginary)),
                           mul(coefficient[0], z.imaginary)),
                       mul(coefficient[1], z.real));
      bins[k] = packBin(floorShift(accR, 16), floorShift(accI, 16));
    }

    RankedTensorType resultType = op.getResult().getType();
    Value result =
        rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
    for (int64_t index = 0; index < 17; ++index) {
      Value position = rewriter.create<arith::ConstantIndexOp>(loc, index);
      result = rewriter.create<tensor::InsertOp>(loc, bins[index], result, position);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

class RfftSplitOpLowering final : public OpConversionPattern<ondrix::ir::RfftSplitOp> {
public:
  using OpConversionPattern<ondrix::ir::RfftSplitOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ir::RfftSplitOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();
    IntegerType i32 = rewriter.getI32Type();
    IntegerType i64 = rewriter.getIntegerType(64);
    int64_t extent = op.getInput().getType().getDimSize(0);

    // The one toward-negative site. Its scale must stay attribute-identical
    // to the scaled saturating subtract the OrtumCore emulation emits, so a
    // later selection pass can still recognize this site.
    auto floorHalveSaturating = [&](Value lhs, Value rhs) -> Value {
      auto scale = ondrix::ondsp::ScaleAttr::get(context, /*preShiftLeft=*/0,
                                                 /*postShiftRight=*/1,
                                                 ondrix::ondsp::RoundingMode::TowardNegative,
                                                 ondrix::ondsp::OverflowMode::Saturate, i32);
      return rewriter.create<ondrix::ondsp::SubShiftOp>(loc, i32, lhs, rhs, scale);
    };
    // Every other rounding decision. The carrier never narrows here, so the
    // scale's overflow mode is unreachable.
    auto truncateShift = [&](Value value, unsigned shift, IntegerType carrier) -> Value {
      auto scale = ondrix::ondsp::ScaleAttr::get(context, /*preShiftLeft=*/0, shift,
                                                 ondrix::ondsp::RoundingMode::TowardZero,
                                                 ondrix::ondsp::OverflowMode::Wrap, carrier);
      return rewriter.create<ondrix::ondsp::RoundShiftOp>(loc, carrier, value, scale);
    };
    auto add = [&](Value lhs, Value rhs) -> Value {
      return rewriter.create<arith::AddIOp>(loc, lhs, rhs);
    };
    auto sub = [&](Value lhs, Value rhs) -> Value {
      return rewriter.create<arith::SubIOp>(loc, lhs, rhs);
    };
    auto widen = [&](Value value) -> Value {
      return rewriter.create<arith::ExtSIOp>(loc, i64, value);
    };
    auto constant64 = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantIntOp>(loc, value, 64);
    };
    auto scaleProduct = [&](int64_t coefficient, Value value) -> Value {
      Value product = rewriter.create<arith::MulIOp>(loc, constant64(coefficient), value);
      return truncateShift(product, 30, i64);
    };
    // The combines are exact in i64 and the halved result provably fits i32,
    // so this boundary truncates without a clamp.
    auto combine = [&](Value value) -> Value {
      return rewriter.create<arith::TruncIOp>(loc, i32, truncateShift(value, 1, i64));
    };

    Value shift32 = rewriter.create<arith::ConstantIntOp>(loc, 32, 64);
    auto unpack = [&](int64_t index) -> SplitComplexValue {
      Value position = rewriter.create<arith::ConstantIndexOp>(loc, index);
      Value packed = rewriter.create<tensor::ExtractOp>(loc, adaptor.getInput(), position);
      Value real = rewriter.create<arith::TruncIOp>(loc, i32, packed);
      Value high = rewriter.create<arith::ShRUIOp>(loc, packed, shift32);
      return {real, rewriter.create<arith::TruncIOp>(loc, i32, high)};
    };
    auto packBin = [&](Value real, Value imaginary) -> Value {
      Value realBits = rewriter.create<arith::ExtUIOp>(loc, i64, real);
      Value imaginaryBits = rewriter.create<arith::ExtUIOp>(loc, i64, imaginary);
      Value shifted = rewriter.create<arith::ShLIOp>(loc, imaginaryBits, shift32);
      return rewriter.create<arith::OrIOp>(loc, shifted, realBits);
    };

    int64_t half = extent / 2;
    SmallVector<Value> bins(extent);
    SplitComplexValue dc = unpack(0);
    bins[0] = packBin(truncateShift(add(dc.real, dc.imaginary), 1, i32),
                      rewriter.create<arith::ConstantIntOp>(loc, 0, 32));

    for (int64_t k = 1; k < half; ++k) {
      std::optional<ondrix::Q30SplitTwiddle> twiddle = ondrix::getQ30SplitTwiddle(extent, k);
      assert(twiddle && "the verified split extent must have a frozen twiddle pair");
      SplitComplexValue x = unpack(k);
      SplitComplexValue mirror = unpack(extent - k);
      Value ar = widen(floorHalveSaturating(x.real, mirror.real));
      Value sr = widen(truncateShift(add(x.real, mirror.real), 1, i32));
      Value ai = widen(truncateShift(sub(x.imaginary, mirror.imaginary), 1, i32));
      Value si = widen(truncateShift(add(x.imaginary, mirror.imaginary), 1, i32));
      Value p1 = scaleProduct(twiddle->sine, ar);
      Value p2 = scaleProduct(twiddle->cosine, si);
      Value p3 = scaleProduct(twiddle->sine, si);
      Value p4 = scaleProduct(twiddle->cosine, ar);
      bins[k] = packBin(combine(add(sub(sr, p1), p2)), combine(sub(sub(ai, p3), p4)));
      bins[extent - k] =
          packBin(combine(sub(add(sr, p1), p2)), combine(sub(sub(sub(constant64(0), ai), p3), p4)));
    }

    // The self-paired bin has ar and ai exactly zero, so the p1 and p4 terms
    // are absent rather than folded away.
    std::optional<ondrix::Q30SplitTwiddle> selfTwiddle = ondrix::getQ30SplitTwiddle(extent, half);
    assert(selfTwiddle && "the verified split extent must have a frozen twiddle pair");
    SplitComplexValue self = unpack(half);
    Value selfSr = widen(truncateShift(add(self.real, self.real), 1, i32));
    Value selfSi = widen(truncateShift(add(self.imaginary, self.imaginary), 1, i32));
    Value selfP2 = scaleProduct(selfTwiddle->cosine, selfSi);
    Value selfP3 = scaleProduct(selfTwiddle->sine, selfSi);
    bins[half] = packBin(combine(add(selfSr, selfP2)), combine(sub(constant64(0), selfP3)));

    RankedTensorType resultType = op.getResult().getType();
    Value result =
        rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
    for (int64_t index = 0; index < extent; ++index) {
      Value position = rewriter.create<arith::ConstantIndexOp>(loc, index);
      result = rewriter.create<tensor::InsertOp>(loc, bins[index], result, position);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

class DctOpLowering final : public OpConversionPattern<ondrix::ir::DctOp> {
public:
  using OpConversionPattern<ondrix::ir::DctOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ir::DctOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    int64_t extent = op.getInput().getType().getDimSize(0);
    if (auto fp = dyn_cast<ondrix::ondsp::FpAttr>(op.getInputNumeric())) {
      Type element = fp.getFormat();
      RankedTensorType outputType = op.getResult().getType();
      Value output = rewriter.create<tensor::EmptyOp>(loc, outputType.getShape(), element);
      for (int64_t k = 0; k < extent; ++k) {
        Value sum;
        for (int64_t n = 0; n < extent; ++n) {
          Value position = rewriter.create<arith::ConstantIndexOp>(loc, n);
          Value value = rewriter.create<tensor::ExtractOp>(loc, adaptor.getInput(), position);
          Value coefficient = rewriter.create<arith::ConstantOp>(
              loc, rewriter.getFloatAttr(element, ondrix::getDctCoefficientF32(extent, k, n)));
          sum = sum ? createFpAccumulatorUpdate(loc, value, coefficient, sum, fp, rewriter)
                    : createFpMultiply(loc, value, coefficient, rewriter);
        }
        Value position = rewriter.create<arith::ConstantIndexOp>(loc, k);
        output = rewriter.create<tensor::InsertOp>(loc, sum, output, position);
      }
      rewriter.replaceOp(op, output);
      return success();
    }
    if (!ondrix::hasAdmissibleDctCoefficients(extent))
      return rewriter.notifyMatchFailure(op, "DCT coefficient quantization is not tie-guard "
                                             "admissible");
    IntegerType i64 = rewriter.getIntegerType(64);
    unsigned stageCount = llvm::Log2_64(extent);
    ondrix::ondsp::ScaleAttr scale =
        getNearestEvenSaturatingShift(rewriter.getContext(), 16 + stageCount);

    SmallVector<Value> inputs;
    inputs.reserve(extent);
    for (int64_t index = 0; index < extent; ++index) {
      Value position = rewriter.create<arith::ConstantIndexOp>(loc, index);
      Value element = rewriter.create<tensor::ExtractOp>(loc, adaptor.getInput(), position);
      inputs.push_back(rewriter.create<arith::ExtSIOp>(loc, i64, element));
    }

    RankedTensorType resultType = op.getResult().getType();
    Value result =
        rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
    for (int64_t k = 0; k < extent; ++k) {
      // Products and the sum are exact in i64 (|sum| <= N * 2^30 < 2^36),
      // so this reduction has no observable association; the single
      // boundary is the final round_shift.
      Value sum;
      for (int64_t n = 0; n < extent; ++n) {
        int64_t coefficient = *ondrix::getDctCoefficientQ15(extent, k, n);
        Value constant =
            rewriter.create<arith::ConstantOp>(loc, i64, rewriter.getIntegerAttr(i64, coefficient));
        Value product = rewriter.create<arith::MulIOp>(loc, inputs[n], constant);
        sum = sum ? rewriter.create<arith::AddIOp>(loc, sum, product).getResult() : product;
      }
      Value exported =
          rewriter.create<ondrix::ondsp::RoundShiftOp>(loc, rewriter.getI16Type(), sum, scale);
      Value position = rewriter.create<arith::ConstantIndexOp>(loc, k);
      result = rewriter.create<tensor::InsertOp>(loc, exported, result, position);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

class CxMagnitudeOpLowering final : public OpConversionPattern<ondrix::ir::CxMagnitudeOp> {
public:
  using OpConversionPattern<ondrix::ir::CxMagnitudeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ir::CxMagnitudeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    IntegerType i16 = rewriter.getI16Type();
    IntegerType i64 = rewriter.getIntegerType(64);
    auto roundingAttr =
        ondrix::ondsp::RoundingModeAttr::get(rewriter.getContext(), op.getRounding());

    int64_t extent = op.getInput().getType().getDimSize(0);
    Value shift = rewriter.create<arith::ConstantIntOp>(loc, 16, 32);
    RankedTensorType resultType = op.getResult().getType();
    Value result =
        rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
    for (int64_t index = 0; index < extent; ++index) {
      Value position = rewriter.create<arith::ConstantIndexOp>(loc, index);
      Value packed = rewriter.create<tensor::ExtractOp>(loc, adaptor.getInput(), position);
      Value real = rewriter.create<arith::TruncIOp>(loc, i16, packed);
      Value high = rewriter.create<arith::ShRUIOp>(loc, packed, shift);
      Value imaginary = rewriter.create<arith::TruncIOp>(loc, i16, high);
      Value realWide = rewriter.create<arith::ExtSIOp>(loc, i64, real);
      Value imaginaryWide = rewriter.create<arith::ExtSIOp>(loc, i64, imaginary);
      Value realSquare = rewriter.create<arith::MulIOp>(loc, realWide, realWide);
      Value imaginarySquare = rewriter.create<arith::MulIOp>(loc, imaginaryWide, imaginaryWide);
      Value sum = rewriter.create<arith::AddIOp>(loc, realSquare, imaginarySquare);
      Value magnitude = rewriter.create<ondrix::ondsp::SqrtFixedOp>(loc, i16, sum, roundingAttr);
      result = rewriter.create<tensor::InsertOp>(loc, magnitude, result, position);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void ondrix::conversion::populateOndrixSpectralLoweringPatterns(RewritePatternSet &patterns,
                                                                bool vectorizeStaticCfft,
                                                                bool fftLoops) {
  MLIRContext *context = patterns.getContext();
  patterns.add<ButterflyOpLowering, RfftRadix4SplitOpLowering, RfftSplitOpLowering, DctOpLowering,
               CxMagnitudeOpLowering>(context);
  patterns.add<CfftOpLowering, RfftOpLowering, IrfftOpLowering>(context, vectorizeStaticCfft,
                                                                fftLoops);
}

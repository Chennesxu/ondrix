#include "OndrixToOndspCommon.h"

#include "ondrix/Dialect/ondrix/IR/OndrixDialect.h"
#include "ondrix/Dialect/ondrix/IR/OndrixOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"
#include "ondrix/Support/GuardedQ15Quantization.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <optional>
#include <tuple>
#include <utility>

using namespace mlir;
using namespace ondrix::conversion;

namespace {

static void assertValidSosSectionShape(Location loc, Value coefficientSections, Value scaleSections,
                                       Value stateSections, Value zero, OpBuilder &builder) {
  Value hasSections =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, coefficientSections, zero);
  builder.create<cf::AssertOp>(loc, hasSections,
                               builder.getStringAttr("SOS filter requires at least one section"));
  Value scalesMatch = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                    coefficientSections, scaleSections);
  Value stateMatches = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                     coefficientSections, stateSections);
  Value sectionsMatch = builder.create<arith::AndIOp>(loc, scalesMatch, stateMatches);
  builder.create<cf::AssertOp>(
      loc, sectionsMatch,
      builder.getStringAttr("SOS coefficient, scale, and state section counts must match"));
}

class SosFilterTdf2OpLowering final : public OpConversionPattern<ondrix::ir::SosFilterTdf2Op> {
public:
  using OpConversionPattern<ondrix::ir::SosFilterTdf2Op>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ir::SosFilterTdf2Op op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value inputLength = rewriter.create<tensor::DimOp>(loc, adaptor.getInput(), zero);
    Value coefficientSections = rewriter.create<tensor::DimOp>(loc, adaptor.getCoeffs(), zero);
    Value scaleSections = rewriter.create<tensor::DimOp>(loc, adaptor.getScales(), zero);
    Value stateSections = rewriter.create<tensor::DimOp>(loc, adaptor.getState(), zero);

    assertValidSosSectionShape(loc, coefficientSections, scaleSections, stateSections, zero,
                               rewriter);

    Value emptyOutput = createEmptyTensor(loc, op.getOutput().getType(), inputLength, rewriter);
    auto numeric = cast<ondrix::ondsp::FpAttr>(op.getNumeric());
    Value coefficientZero = zero;
    Value coefficientOne = one;
    Value coefficientTwo = rewriter.create<arith::ConstantIndexOp>(loc, 2);
    Value coefficientThree = rewriter.create<arith::ConstantIndexOp>(loc, 3);
    Value coefficientFour = rewriter.create<arith::ConstantIndexOp>(loc, 4);

    auto sampleLoop = rewriter.create<scf::ForOp>(
        loc, zero, inputLength, one, ValueRange{emptyOutput, adaptor.getState()},
        [&](OpBuilder &builder, Location sampleLoc, Value sampleIndex, ValueRange sampleArgs) {
          Value sample = builder.create<tensor::ExtractOp>(sampleLoc, adaptor.getInput(),
                                                           ValueRange{sampleIndex});
          auto sectionLoop = builder.create<scf::ForOp>(
              sampleLoc, zero, coefficientSections, one, ValueRange{sample, sampleArgs[1]},
              [&](OpBuilder &sectionBuilder, Location sectionLoc, Value section,
                  ValueRange sectionArgs) {
                auto extractCoefficient = [&](Value column) {
                  return sectionBuilder.create<tensor::ExtractOp>(sectionLoc, adaptor.getCoeffs(),
                                                                  ValueRange{section, column});
                };
                Value scale = sectionBuilder.create<tensor::ExtractOp>(
                    sectionLoc, adaptor.getScales(), ValueRange{section});
                Value b0 = extractCoefficient(coefficientZero);
                Value b1 = extractCoefficient(coefficientOne);
                Value b2 = extractCoefficient(coefficientTwo);
                Value a1 = extractCoefficient(coefficientThree);
                Value a2 = extractCoefficient(coefficientFour);
                Value z1 = sectionBuilder.create<tensor::ExtractOp>(
                    sectionLoc, sectionArgs[1], ValueRange{section, coefficientZero});
                Value z2 = sectionBuilder.create<tensor::ExtractOp>(
                    sectionLoc, sectionArgs[1], ValueRange{section, coefficientOne});

                Value scaled = createFpMultiply(sectionLoc, sectionArgs[0], scale, sectionBuilder);
                Value output =
                    createFpAccumulatorUpdate(sectionLoc, scaled, b0, z1, numeric, sectionBuilder);
                Value feedback1 = createFpMultiply(sectionLoc, output, a1, sectionBuilder);
                Value firstTerm = createFpAccumulatorUpdate(sectionLoc, scaled, b1, feedback1,
                                                            numeric, sectionBuilder);
                Value nextZ1 = createFpAdd(sectionLoc, z2, firstTerm, sectionBuilder);
                Value feedback2 = createFpMultiply(sectionLoc, output, a2, sectionBuilder);
                Value nextZ2 = createFpAccumulatorUpdate(sectionLoc, scaled, b2, feedback2, numeric,
                                                         sectionBuilder);
                Value stateWithZ1 = sectionBuilder.create<tensor::InsertOp>(
                    sectionLoc, nextZ1, sectionArgs[1], ValueRange{section, coefficientZero});
                Value nextState = sectionBuilder.create<tensor::InsertOp>(
                    sectionLoc, nextZ2, stateWithZ1, ValueRange{section, coefficientOne});
                sectionBuilder.create<scf::YieldOp>(sectionLoc, ValueRange{output, nextState});
              });
          Value nextOutput = builder.create<tensor::InsertOp>(
              sampleLoc, sectionLoop.getResult(0), sampleArgs[0], ValueRange{sampleIndex});
          builder.create<scf::YieldOp>(sampleLoc, ValueRange{nextOutput, sectionLoop.getResult(1)});
        });

    rewriter.replaceOp(op, sampleLoop.getResults());
    return success();
  }
};

/// Longest cascade emitted straight-line; the measured argument is the
/// `convert-ondrix-to-ondsp` pass description's.
constexpr int64_t kMaxStraightLineSections = 8;

/// The cascade length, when all three sectioned operands carry the same static
/// count. A count only the coefficients declare is not one: the loop bound and
/// the runtime assert are what keep the other two operands in range.
static std::optional<int64_t> getStraightLineSectionCount(ondrix::ir::SosFilterDf2FixedOp op) {
  int64_t sections = op.getCoeffs().getType().getDimSize(0);
  if (ShapedType::isDynamic(sections) || sections > kMaxStraightLineSections)
    return std::nullopt;
  if (op.getScales().getType().getDimSize(0) != sections ||
      op.getState().getType().getDimSize(0) != sections)
    return std::nullopt;
  return sections;
}

class SosFilterDf2FixedOpLowering final
    : public OpConversionPattern<ondrix::ir::SosFilterDf2FixedOp> {
public:
  using OpConversionPattern<ondrix::ir::SosFilterDf2FixedOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ir::SosFilterDf2FixedOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value two = rewriter.create<arith::ConstantIndexOp>(loc, 2);
    Value three = rewriter.create<arith::ConstantIndexOp>(loc, 3);
    Value four = rewriter.create<arith::ConstantIndexOp>(loc, 4);
    Value inputLength = rewriter.create<tensor::DimOp>(loc, adaptor.getInput(), zero);
    Value coefficientSections = rewriter.create<tensor::DimOp>(loc, adaptor.getCoeffs(), zero);
    Value scaleSections = rewriter.create<tensor::DimOp>(loc, adaptor.getScales(), zero);
    Value stateSections = rewriter.create<tensor::DimOp>(loc, adaptor.getState(), zero);

    assertValidSosSectionShape(loc, coefficientSections, scaleSections, stateSections, zero,
                               rewriter);

    Value emptyOutput = createEmptyTensor(loc, op.getOutput().getType(), inputLength, rewriter);
    auto createMac = [&](OpBuilder &builder, Location updateLoc, Value accumulator, Value lhs,
                         Value rhs) {
      return builder.create<ondrix::ondsp::MacOp>(updateLoc, op.getAccumulator(), accumulator, lhs,
                                                  rhs, op.getNumeric(), op.getProduct());
    };

    auto emitSection = [&](OpBuilder &sectionBuilder, Location sectionLoc, Value section,
                           Value sectionInput, Value state) -> std::pair<Value, Value> {
      auto extractCoefficient = [&](Value column) {
        return sectionBuilder.create<tensor::ExtractOp>(sectionLoc, adaptor.getCoeffs(),
                                                        ValueRange{section, column});
      };
      Value scale = sectionBuilder.create<tensor::ExtractOp>(sectionLoc, adaptor.getScales(),
                                                             ValueRange{section});
      Value b0 = extractCoefficient(zero);
      Value b1 = extractCoefficient(one);
      Value b2 = extractCoefficient(two);
      Value a1 = extractCoefficient(three);
      Value a2 = extractCoefficient(four);
      Value d1 =
          sectionBuilder.create<tensor::ExtractOp>(sectionLoc, state, ValueRange{section, zero});
      Value d2 =
          sectionBuilder.create<tensor::ExtractOp>(sectionLoc, state, ValueRange{section, one});

      Value stateAccumulator =
          sectionBuilder.create<ondrix::ondsp::AccZeroOp>(sectionLoc, op.getAccumulator());
      stateAccumulator =
          createMac(sectionBuilder, sectionLoc, stateAccumulator, sectionInput, scale);
      stateAccumulator = createMac(sectionBuilder, sectionLoc, stateAccumulator, d1, a1);
      stateAccumulator = createMac(sectionBuilder, sectionLoc, stateAccumulator, d2, a2);
      Value nextD1 = sectionBuilder.create<ondrix::ondsp::AccExportOp>(
          sectionLoc, op.getNumeric().getStorage(), stateAccumulator, op.getNumeric(),
          op.getStateRounding(), op.getStateOverflow());

      Value outputAccumulator =
          sectionBuilder.create<ondrix::ondsp::AccZeroOp>(sectionLoc, op.getAccumulator());
      outputAccumulator = createMac(sectionBuilder, sectionLoc, outputAccumulator, nextD1, b0);
      outputAccumulator = createMac(sectionBuilder, sectionLoc, outputAccumulator, d1, b1);
      outputAccumulator = createMac(sectionBuilder, sectionLoc, outputAccumulator, d2, b2);
      Value output = sectionBuilder.create<ondrix::ondsp::AccExportOp>(
          sectionLoc, op.getNumeric().getStorage(), outputAccumulator, op.getNumeric(),
          op.getOutputRounding(), op.getOutputOverflow());

      Value stateWithD1 = sectionBuilder.create<tensor::InsertOp>(sectionLoc, nextD1, state,
                                                                  ValueRange{section, zero});
      Value nextState = sectionBuilder.create<tensor::InsertOp>(sectionLoc, d1, stateWithD1,
                                                                ValueRange{section, one});
      return {output, nextState};
    };

    std::optional<int64_t> straightLineSections = getStraightLineSectionCount(op);
    auto emitCascade = [&](OpBuilder &builder, Location sampleLoc, Value sample,
                           Value state) -> std::pair<Value, Value> {
      if (straightLineSections) {
        Value carried = sample;
        for (int64_t index = 0; index < *straightLineSections; ++index) {
          Value section = builder.create<arith::ConstantIndexOp>(sampleLoc, index);
          std::tie(carried, state) = emitSection(builder, sampleLoc, section, carried, state);
        }
        return {carried, state};
      }
      auto sectionLoop = builder.create<scf::ForOp>(
          sampleLoc, zero, coefficientSections, one, ValueRange{sample, state},
          [&](OpBuilder &sectionBuilder, Location sectionLoc, Value section,
              ValueRange sectionArgs) {
            auto [output, nextState] =
                emitSection(sectionBuilder, sectionLoc, section, sectionArgs[0], sectionArgs[1]);
            sectionBuilder.create<scf::YieldOp>(sectionLoc, ValueRange{output, nextState});
          });
      return {sectionLoop.getResult(0), sectionLoop.getResult(1)};
    };

    auto sampleLoop = rewriter.create<scf::ForOp>(
        loc, zero, inputLength, one, ValueRange{emptyOutput, adaptor.getState()},
        [&](OpBuilder &builder, Location sampleLoc, Value sampleIndex, ValueRange sampleArgs) {
          Value sample = builder.create<tensor::ExtractOp>(sampleLoc, adaptor.getInput(),
                                                           ValueRange{sampleIndex});
          auto [cascaded, nextState] = emitCascade(builder, sampleLoc, sample, sampleArgs[1]);
          Value nextOutput = builder.create<tensor::InsertOp>(sampleLoc, cascaded, sampleArgs[0],
                                                              ValueRange{sampleIndex});
          builder.create<scf::YieldOp>(sampleLoc, ValueRange{nextOutput, nextState});
        });

    rewriter.replaceOp(op, sampleLoop.getResults());
    return success();
  }
};

class CicDecimateOpLowering final : public OpConversionPattern<ondrix::ir::CicDecimateOp> {
public:
  using OpConversionPattern<ondrix::ir::CicDecimateOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ir::CicDecimateOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();
    int64_t stages = op.getStages();
    int64_t rate = op.getRate();
    int64_t delay = op.getDelay();
    int64_t growth = op.getGrowthBits();
    int64_t outputs = op.getResult().getType().getDimSize(0);
    auto carrier = IntegerType::get(context, 16 + growth);

    // Every state combine is the declared-overflow boundary at the carrier
    // width; only the export rounds. The rounding field of the state scale
    // is vacuous at post_shift_right = 0.
    auto stateScale = ondrix::ondsp::ScaleAttr::get(
        context, /*preShiftLeft=*/0, /*postShiftRight=*/0,
        ondrix::ondsp::RoundingMode::TowardNegative, op.getOverflow(), carrier);
    auto exportScale = ondrix::ondsp::ScaleAttr::get(
        context, /*preShiftLeft=*/0, /*postShiftRight=*/unsigned(growth), op.getRounding(),
        ondrix::ondsp::OverflowMode::Saturate, rewriter.getI16Type());

    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value rateValue = rewriter.create<arith::ConstantIndexOp>(loc, rate);
    Value outputCount = rewriter.create<arith::ConstantIndexOp>(loc, outputs);
    Value carrierZero = rewriter.create<arith::ConstantIntOp>(loc, 0, carrier);

    // Integrator bank at the input rate, then phase R-1 selection. The state
    // vector rides the outer loop so the rate change costs no buffer.
    RankedTensorType decimatedType = RankedTensorType::get({outputs}, carrier);
    SmallVector<Value> integratorInit(stages, carrierZero);
    integratorInit.push_back(
        rewriter.create<tensor::EmptyOp>(loc, decimatedType.getShape(), carrier));
    auto integrate = rewriter.create<scf::ForOp>(
        loc, zero, outputCount, one, integratorInit,
        [&](OpBuilder &builder, Location loc, Value block, ValueRange outer) {
          Value base = builder.create<arith::MulIOp>(loc, block, rateValue);
          auto phases = builder.create<scf::ForOp>(
              loc, zero, rateValue, one, outer.drop_back(),
              [&](OpBuilder &builder, Location loc, Value phase, ValueRange state) {
                Value position = builder.create<arith::AddIOp>(loc, base, phase);
                Value sample = builder.create<tensor::ExtractOp>(loc, adaptor.getInput(), position);
                Value carried = builder.create<arith::ExtSIOp>(loc, carrier, sample);
                SmallVector<Value> next;
                for (Value stage : state) {
                  carried = builder.create<ondrix::ondsp::AddShiftOp>(loc, carrier, stage, carried,
                                                                      stateScale);
                  next.push_back(carried);
                }
                builder.create<scf::YieldOp>(loc, next);
              });
          Value inserted = builder.create<tensor::InsertOp>(loc, phases.getResults().back(),
                                                            outer.back(), block);
          SmallVector<Value> next(phases.getResults().begin(), phases.getResults().end());
          next.push_back(inserted);
          builder.create<scf::YieldOp>(loc, next);
        });

    // Comb bank at the output rate. Each stage keeps its own M-deep delay
    // line as loop-carried values, so the differencing needs no history
    // tensor and the zero prehistory is the initial state.
    RankedTensorType resultType = op.getResult().getType();
    SmallVector<Value> combInit(stages * delay, carrierZero);
    combInit.push_back(
        rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType()));
    Value decimated = integrate.getResults().back();
    auto comb = rewriter.create<scf::ForOp>(
        loc, zero, outputCount, one, combInit,
        [&](OpBuilder &builder, Location loc, Value index, ValueRange state) {
          Value carried = builder.create<tensor::ExtractOp>(loc, decimated, index);
          SmallVector<Value> next;
          for (int64_t stage = 0; stage < stages; ++stage) {
            // line[t] holds this stage's input delayed by t+1 outputs, so
            // line.back() is the c[j-1, m-M] the difference needs.
            ValueRange line = state.slice(stage * delay, delay);
            Value differenced = builder.create<ondrix::ondsp::SubShiftOp>(loc, carrier, carried,
                                                                          line.back(), stateScale);
            next.push_back(carried);
            for (int64_t tap = 1; tap < delay; ++tap)
              next.push_back(line[tap - 1]);
            carried = differenced;
          }
          Value exported = builder.create<ondrix::ondsp::RoundShiftOp>(loc, builder.getI16Type(),
                                                                       carried, exportScale);
          next.push_back(builder.create<tensor::InsertOp>(loc, exported, state.back(), index));
          builder.create<scf::YieldOp>(loc, next);
        });
    rewriter.replaceOp(op, comb.getResults().back());
    return success();
  }
};

// cos(2*pi*k/N) with the quarter-turn angles evaluated exactly. binary64
// cannot represent pi/2, so libm returns about 1e-16 where the exact cosine
// is zero, and bin N/4 would otherwise not name the bin it says it does.
static double turnCosine(int64_t bin, int64_t extent) {
  constexpr double kTwoPi = 6.28318530717958647692528676655900577;
  if (4 * bin % extent == 0) {
    static constexpr double kQuarterTurns[4] = {1.0, 0.0, -1.0, 0.0};
    return kQuarterTurns[(4 * bin / extent) % 4];
  }
  return std::cos(kTwoPi * static_cast<double>(bin) / static_cast<double>(extent));
}

class GoertzelOpLowering final : public OpConversionPattern<ondrix::ir::GoertzelOp> {
public:
  using OpConversionPattern<ondrix::ir::GoertzelOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ir::GoertzelOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    int64_t extent = op.getInput().getType().getDimSize(0);
    double cosine = turnCosine(op.getBin(), extent);
    if (auto fp = dyn_cast<ondrix::ondsp::FpAttr>(op.getNumeric()))
      return rewriteFloatingPoint(op, adaptor, fp, extent, cosine, rewriter);
    // The fixed recursion coefficient is one tie-guarded compile-time cosine
    // (the same guarded quantizer as every generated table); inadmissible
    // bins fail closed.
    std::optional<ondrix::GuardedQ15Value> coefficient = ondrix::quantizeGuardedQ15(cosine);
    if (!coefficient)
      return rewriter.notifyMatchFailure(op, "bin coefficient is not tie-guard admissible");
    IntegerType i16 = rewriter.getI16Type();
    IntegerType i32 = rewriter.getIntegerType(32);
    IntegerType i64 = rewriter.getIntegerType(64);
    Attribute numeric = op.getNumeric();
    ondrix::ondsp::ScaleAttr scale = getNearestEvenSaturatingShift(rewriter.getContext(), 15);
    Value doubledCoefficient =
        rewriter.create<arith::ConstantIntOp>(loc, 2 * int64_t(coefficient->value), 64);
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value extentValue = rewriter.create<arith::ConstantIndexOp>(loc, extent);
    Value zero16 = rewriter.create<arith::ConstantIntOp>(loc, 0, 16);

    // m = sat_i16(rhe(2*c*s1 / 2^15)) — the per-step product boundary.
    auto stepProduct = [&](OpBuilder &builder, Location loc, Value s1) -> Value {
      Value wide = builder.create<arith::ExtSIOp>(loc, i64, s1);
      Value product = builder.create<arith::MulIOp>(loc, doubledCoefficient, wide);
      return builder.create<ondrix::ondsp::RoundShiftOp>(loc, i16, product, scale);
    };

    auto sampleLoop = rewriter.create<scf::ForOp>(
        loc, zero, extentValue, one, ValueRange{zero16, zero16},
        [&](OpBuilder &builder, Location loc, Value sample, ValueRange states) {
          Value s1 = states[0];
          Value s2 = states[1];
          Value m = stepProduct(builder, loc, s1);
          Value x = builder.create<tensor::ExtractOp>(loc, adaptor.getInput(), sample);
          Value xWide = builder.create<arith::ExtSIOp>(loc, i32, x);
          Value mWide = builder.create<arith::ExtSIOp>(loc, i32, m);
          Value s2Wide = builder.create<arith::ExtSIOp>(loc, i32, s2);
          Value sum = builder.create<arith::AddIOp>(loc, xWide, mWide);
          Value combined = builder.create<arith::SubIOp>(loc, sum, s2Wide);
          Value s0 = builder.create<ondrix::ondsp::SatCastOp>(loc, i16, combined, numeric);
          builder.create<scf::YieldOp>(loc, ValueRange{s0, s1});
        });
    Value s1 = sampleLoop.getResult(0);
    Value s2 = sampleLoop.getResult(1);
    Value mFinal = stepProduct(rewriter, loc, s1);

    // energy = s1^2 + s2^2 - m*s2, exact in i64; no further boundary.
    Value s1Wide = rewriter.create<arith::ExtSIOp>(loc, i64, s1);
    Value s2Wide = rewriter.create<arith::ExtSIOp>(loc, i64, s2);
    Value mWide = rewriter.create<arith::ExtSIOp>(loc, i64, mFinal);
    Value s1Square = rewriter.create<arith::MulIOp>(loc, s1Wide, s1Wide);
    Value s2Square = rewriter.create<arith::MulIOp>(loc, s2Wide, s2Wide);
    Value cross = rewriter.create<arith::MulIOp>(loc, mWide, s2Wide);
    Value sum = rewriter.create<arith::AddIOp>(loc, s1Square, s2Square);
    Value energy = rewriter.create<arith::SubIOp>(loc, sum, cross);

    RankedTensorType resultType = op.getEnergy().getType();
    Value empty =
        rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
    Value result = rewriter.create<tensor::InsertOp>(loc, energy, empty, zero);
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  // Doubling the rounded cosine is exact, so `c2` carries no boundary of its
  // own. Everything but the coefficient product and its input addition is
  // built unflagged, which is what bounds the derivable set the operation
  // description declares.
  static LogicalResult rewriteFloatingPoint(ondrix::ir::GoertzelOp op, OpAdaptor adaptor,
                                            ondrix::ondsp::FpAttr numeric, int64_t extent,
                                            double cosine, ConversionPatternRewriter &rewriter) {
    Location loc = op.getLoc();
    Type element = numeric.getFormat();
    Value doubledCoefficient = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(element, 2.0 * static_cast<double>(static_cast<float>(cosine))));
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value extentValue = rewriter.create<arith::ConstantIndexOp>(loc, extent);
    Value seed = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(element));
    auto sampleLoop = rewriter.create<scf::ForOp>(
        loc, zero, extentValue, one, ValueRange{seed, seed},
        [&](OpBuilder &builder, Location loc, Value sample, ValueRange states) {
          Value x = builder.create<tensor::ExtractOp>(loc, adaptor.getInput(), sample);
          Value combined =
              createFpAccumulatorUpdate(loc, doubledCoefficient, states[0], x, numeric, builder);
          Value next = builder.create<arith::SubFOp>(loc, combined, states[1]);
          builder.create<scf::YieldOp>(loc, ValueRange{next, states[0]});
        });
    Value s1 = sampleLoop.getResult(0);
    Value s2 = sampleLoop.getResult(1);
    Value m = rewriter.create<arith::MulFOp>(loc, doubledCoefficient, s1);
    Value s1Square = rewriter.create<arith::MulFOp>(loc, s1, s1);
    Value s2Square = rewriter.create<arith::MulFOp>(loc, s2, s2);
    Value cross = rewriter.create<arith::MulFOp>(loc, m, s2);
    Value sum = rewriter.create<arith::AddFOp>(loc, s1Square, s2Square);
    Value energy = rewriter.create<arith::SubFOp>(loc, sum, cross);

    RankedTensorType resultType = op.getEnergy().getType();
    Value empty =
        rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
    rewriter.replaceOp(op, rewriter.create<tensor::InsertOp>(loc, energy, empty, zero).getResult());
    return success();
  }
};

class LmsOpLowering final : public OpConversionPattern<ondrix::ir::LmsOp> {
public:
  using OpConversionPattern<ondrix::ir::LmsOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ir::LmsOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    int64_t samples = op.getInput().getType().getDimSize(0);
    int64_t taps = op.getWeights().getType().getDimSize(0);
    IntegerType i16 = rewriter.getI16Type();
    IntegerType i32 = rewriter.getIntegerType(32);
    IntegerType i64 = rewriter.getIntegerType(64);
    Attribute numeric = op.getNumeric();
    auto fp = dyn_cast<ondrix::ondsp::FpAttr>(numeric);
    Type element = fp ? Type(fp.getFormat()) : Type(i16);
    // Every fixed rounding boundary of the recursion is one nearest-even
    // saturating round_shift by 15; the error and weight updates use
    // explicit saturating casts. The f32 profile has no boundary at any of
    // them. The whole recursion is loop-form either way: the weight state
    // flows sample to sample as an iter_arg.
    ondrix::ondsp::ScaleAttr scale;
    Value mu;
    if (fp) {
      mu = rewriter.create<arith::ConstantOp>(loc, op.getFpStepSizeAttr());
    } else {
      scale = getNearestEvenSaturatingShift(rewriter.getContext(), 15);
      mu = rewriter.create<arith::ConstantIntOp>(loc, op.getStepSizeAttr().getInt(), 64);
    }
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value sampleCount = rewriter.create<arith::ConstantIndexOp>(loc, samples);
    Value tapCount = rewriter.create<arith::ConstantIndexOp>(loc, taps);
    Value zeroElement = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(element));

    // Zero-prehistory tap fetch: x[n - k], 0 for n < k. The prehistory term
    // is evaluated rather than skipped on both profiles.
    auto guardedInput = [&](OpBuilder &builder, Location loc, Value sample, Value tap) -> Value {
      Value offset = builder.create<arith::SubIOp>(loc, sample, tap);
      Value valid = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, offset, zero);
      Value clamped = builder.create<arith::MaxSIOp>(loc, offset, zero);
      Value value = builder.create<tensor::ExtractOp>(loc, adaptor.getInput(), clamped);
      return builder.create<arith::SelectOp>(loc, valid, value, zeroElement);
    };
    // The same fetch where k <= K - 1 <= n already proves n - k in range.
    auto unguardedInput = [&](OpBuilder &builder, Location loc, Value sample, Value tap) -> Value {
      Value offset = builder.create<arith::SubIOp>(loc, sample, tap);
      return builder.create<tensor::ExtractOp>(loc, adaptor.getInput(), offset);
    };

    // Copy the initial weights into a fresh tensor before adapting: the
    // recursion mutates its weight state per sample, and inserting into
    // the function-argument tensor directly would let bufferization adapt
    // the caller's buffer in place.
    Value weightsEmpty = rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{taps}, element);
    auto copyLoop = rewriter.create<scf::ForOp>(
        loc, zero, tapCount, one, ValueRange{weightsEmpty},
        [&](OpBuilder &builder, Location loc, Value tap, ValueRange iterArgs) {
          Value weight = builder.create<tensor::ExtractOp>(loc, adaptor.getWeights(), tap);
          Value inserted = builder.create<tensor::InsertOp>(loc, weight, iterArgs.front(), tap);
          builder.create<scf::YieldOp>(loc, inserted);
        });

    Value errorsEmpty = rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{samples}, element);

    // The guard can fire only while n < K - 1, so the sample loop splits at
    // min(K - 1, N) into a guarded and an unguarded region running the same
    // body over the same state. An empty region is not emitted.
    int64_t peel = std::min(taps - 1, samples);
    Value peelPoint = zero;
    if (peel == samples)
      peelPoint = sampleCount;
    else if (peel > 0)
      peelPoint = rewriter.create<arith::ConstantIndexOp>(loc, peel);

    auto emitSampleRegions = [&](SampleBody body) -> SmallVector<Value> {
      SmallVector<Value> state{copyLoop.getResult(0), errorsEmpty};
      if (peel > 0) {
        auto guarded = rewriter.create<scf::ForOp>(
            loc, zero, peelPoint, one, state,
            [&](OpBuilder &builder, Location loc, Value sample, ValueRange iterArgs) {
              body(builder, loc, sample, iterArgs, guardedInput);
            });
        state.assign(guarded.getResults().begin(), guarded.getResults().end());
      }
      if (peel < samples) {
        auto settled = rewriter.create<scf::ForOp>(
            loc, peelPoint, sampleCount, one, state,
            [&](OpBuilder &builder, Location loc, Value sample, ValueRange iterArgs) {
              body(builder, loc, sample, iterArgs, unguardedInput);
            });
        state.assign(settled.getResults().begin(), settled.getResults().end());
      }
      return state;
    };

    if (fp) {
      auto fpSample = [&](OpBuilder &builder, Location loc, Value sample, ValueRange iterArgs,
                          TapFetch fetch) {
        Value weights = iterArgs[0];
        auto accLoop = builder.create<scf::ForOp>(
            loc, zero, tapCount, one, ValueRange{zeroElement},
            [&](OpBuilder &builder, Location loc, Value tap, ValueRange accArgs) {
              Value term = fetch(builder, loc, sample, tap);
              Value weight = builder.create<tensor::ExtractOp>(loc, weights, tap);
              Value updated =
                  createFpAccumulatorUpdate(loc, weight, term, accArgs.front(), fp, builder);
              builder.create<scf::YieldOp>(loc, updated);
            });
        Value desired = builder.create<tensor::ExtractOp>(loc, adaptor.getDesired(), sample);
        Value error = builder.create<arith::SubFOp>(loc, desired, accLoop.getResult(0));
        Value nextErrors = builder.create<tensor::InsertOp>(loc, error, iterArgs[1], sample);
        Value step = createFpMultiply(loc, mu, error, builder);

        auto updateLoop = builder.create<scf::ForOp>(
            loc, zero, tapCount, one, ValueRange{weights},
            [&](OpBuilder &builder, Location loc, Value tap, ValueRange updateArgs) {
              Value term = fetch(builder, loc, sample, tap);
              Value weight = builder.create<tensor::ExtractOp>(loc, updateArgs.front(), tap);
              Value updated = createFpAccumulatorUpdate(loc, step, term, weight, fp, builder);
              Value inserted =
                  builder.create<tensor::InsertOp>(loc, updated, updateArgs.front(), tap);
              builder.create<scf::YieldOp>(loc, inserted);
            });
        builder.create<scf::YieldOp>(loc, ValueRange{updateLoop.getResult(0), nextErrors});
      };
      SmallVector<Value> adapted = emitSampleRegions(fpSample);
      rewriter.replaceOp(op, {adapted[1], adapted[0]});
      return success();
    }

    Value zero64 = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
    auto fixedSample = [&](OpBuilder &builder, Location loc, Value sample, ValueRange iterArgs,
                           TapFetch fetch) {
      Value weights = iterArgs[0];
      Value errors = iterArgs[1];

      auto accLoop = builder.create<scf::ForOp>(
          loc, zero, tapCount, one, ValueRange{zero64},
          [&](OpBuilder &builder, Location loc, Value tap, ValueRange accArgs) {
            Value term = fetch(builder, loc, sample, tap);
            Value termWide = builder.create<arith::ExtSIOp>(loc, i64, term);
            Value weight = builder.create<tensor::ExtractOp>(loc, weights, tap);
            Value weightWide = builder.create<arith::ExtSIOp>(loc, i64, weight);
            Value product = builder.create<arith::MulIOp>(loc, weightWide, termWide);
            Value sum = builder.create<arith::AddIOp>(loc, accArgs.front(), product);
            builder.create<scf::YieldOp>(loc, sum);
          });
      Value output =
          builder.create<ondrix::ondsp::RoundShiftOp>(loc, i16, accLoop.getResult(0), scale);

      Value desired = builder.create<tensor::ExtractOp>(loc, adaptor.getDesired(), sample);
      Value desiredWide = builder.create<arith::ExtSIOp>(loc, i32, desired);
      Value outputWide = builder.create<arith::ExtSIOp>(loc, i32, output);
      Value difference = builder.create<arith::SubIOp>(loc, desiredWide, outputWide);
      Value error = builder.create<ondrix::ondsp::SatCastOp>(loc, i16, difference, numeric);
      Value nextErrors = builder.create<tensor::InsertOp>(loc, error, errors, sample);

      Value errorWide = builder.create<arith::ExtSIOp>(loc, i64, error);
      Value stepProduct = builder.create<arith::MulIOp>(loc, mu, errorWide);
      Value step = builder.create<ondrix::ondsp::RoundShiftOp>(loc, i16, stepProduct, scale);
      Value stepWide = builder.create<arith::ExtSIOp>(loc, i64, step);

      auto updateLoop = builder.create<scf::ForOp>(
          loc, zero, tapCount, one, ValueRange{weights},
          [&](OpBuilder &builder, Location loc, Value tap, ValueRange updateArgs) {
            Value term = fetch(builder, loc, sample, tap);
            Value termWide = builder.create<arith::ExtSIOp>(loc, i64, term);
            Value product = builder.create<arith::MulIOp>(loc, stepWide, termWide);
            Value delta = builder.create<ondrix::ondsp::RoundShiftOp>(loc, i16, product, scale);
            Value weight = builder.create<tensor::ExtractOp>(loc, updateArgs.front(), tap);
            Value weightWide = builder.create<arith::ExtSIOp>(loc, i32, weight);
            Value deltaWide = builder.create<arith::ExtSIOp>(loc, i32, delta);
            Value updated = builder.create<arith::AddIOp>(loc, weightWide, deltaWide);
            Value saturated = builder.create<ondrix::ondsp::SatCastOp>(loc, i16, updated, numeric);
            Value inserted =
                builder.create<tensor::InsertOp>(loc, saturated, updateArgs.front(), tap);
            builder.create<scf::YieldOp>(loc, inserted);
          });
      builder.create<scf::YieldOp>(loc, ValueRange{updateLoop.getResult(0), nextErrors});
    };
    SmallVector<Value> adapted = emitSampleRegions(fixedSample);
    rewriter.replaceOp(op, {adapted[1], adapted[0]});
    return success();
  }

private:
  // x[n - k] for one sample and tap, and the per-sample recursion step that
  // both peeled regions share so their arithmetic cannot drift.
  using TapFetch = llvm::function_ref<Value(OpBuilder &, Location, Value, Value)>;
  using SampleBody = llvm::function_ref<void(OpBuilder &, Location, Value, ValueRange, TapFetch)>;
};

} // namespace

void ondrix::conversion::populateOndrixStatefulLoweringPatterns(RewritePatternSet &patterns) {
  patterns.add<SosFilterTdf2OpLowering, SosFilterDf2FixedOpLowering, CicDecimateOpLowering,
               GoertzelOpLowering, LmsOpLowering>(patterns.getContext());
}

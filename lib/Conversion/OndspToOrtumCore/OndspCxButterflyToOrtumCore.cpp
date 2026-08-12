#include "ondrix/Conversion/OndspToOrtumCore/OndspToOrtumCore.h"

#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreDialect.h"
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace ondrix {
#define GEN_PASS_DEF_CONVERTONDSPCXBUTTERFLYTOORTUMCORE
#include "ondrix/Conversion/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

// The packed target rounding inventory; nearest_even deliberately maps to
// nothing so the default profile stays on the generic path.
static std::optional<ondrix::ortumcore::CxRounding>
selectTargetRounding(ondrix::ondsp::RoundingMode mode) {
  switch (mode) {
  case ondrix::ondsp::RoundingMode::TowardNegative:
    return ondrix::ortumcore::CxRounding::TowardNegative;
  case ondrix::ondsp::RoundingMode::NearestTiesPositive:
    return ondrix::ortumcore::CxRounding::NearestTiesPositive;
  default:
    return std::nullopt;
  }
}

static ondrix::ortumcore::CxOverflow selectTargetOverflow(ondrix::ondsp::OverflowMode mode) {
  return mode == ondrix::ondsp::OverflowMode::Wrap ? ondrix::ortumcore::CxOverflow::Wrap
                                                   : ondrix::ortumcore::CxOverflow::Saturate;
}

struct TargetScale {
  ondrix::ortumcore::CxRounding rounding;
  ondrix::ortumcore::CxOverflow overflow;
  int64_t shift;
};

static std::optional<TargetScale> classifyTargetScale(ondrix::ondsp::ScaleAttr scale) {
  std::optional<ondrix::ortumcore::CxRounding> rounding = selectTargetRounding(scale.getRounding());
  if (!rounding || scale.getPreShiftLeft() != 0)
    return std::nullopt;
  auto destination = dyn_cast<IntegerType>(scale.getSaturateTo());
  if (!destination || !destination.isSignless() || destination.getWidth() != 16)
    return std::nullopt;
  return TargetScale{*rounding, selectTargetOverflow(scale.getOverflow()),
                     int64_t(scale.getPostShiftRight())};
}

// The exact conjugate of a packed Q15 constant, or nullopt at the one
// unrepresentable point (imaginary component -32768).
static std::optional<uint32_t> conjugatePackedQ15(uint32_t bits) {
  int32_t imaginary = int16_t(bits >> 16);
  if (imaginary == -32768)
    return std::nullopt;
  return (uint32_t(uint16_t(-imaginary)) << 16) | (bits & 0xFFFF);
}

static void rewriteButterfly(ondrix::ondsp::CxButterflyOp op) {
  if (!op.getA().getType().isSignlessInteger(32) ||
      op.getLayout().getLayout() != ondrix::ondsp::ComplexLayout::PackedI16ImagHiRealLo)
    return;
  std::optional<TargetScale> product = classifyTargetScale(op.getProductScale());
  std::optional<TargetScale> output = classifyTargetScale(op.getOutputScale());
  if (!product || product->shift != 15 || !output || output->shift > 1)
    return;

  auto constant = op.getTwiddle().getDefiningOp<arith::ConstantOp>();
  auto twiddle = constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr();
  if (!twiddle)
    return;
  std::optional<uint32_t> conjugated =
      conjugatePackedQ15(uint32_t(twiddle.getValue().getZExtValue()));
  if (!conjugated)
    return;

  OpBuilder builder(op);
  Location loc = op.getLoc();
  Value conjugate =
      builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(int32_t(*conjugated)));
  Value rotated = builder.create<ondrix::ortumcore::CxMulConjOp>(
      loc, builder.getI32Type(), op.getB(), conjugate, uint64_t(product->shift), product->rounding,
      product->overflow, ondrix::ortumcore::CxLayout::ImagHi);
  auto pair = builder.create<ondrix::ortumcore::CxBflyOp>(
      loc, builder.getI32Type(), builder.getI32Type(), op.getA(), rotated, uint64_t(output->shift),
      output->rounding, output->overflow, ondrix::ortumcore::CxBflyVariant::Plain);
  op.getOut0().replaceAllUsesWith(pair.getOut0());
  op.getOut1().replaceAllUsesWith(pair.getOut1());
  op.erase();
}

class ConvertOndspCxButterflyToOrtumCorePass final
    : public ondrix::impl::ConvertOndspCxButterflyToOrtumCoreBase<
          ConvertOndspCxButterflyToOrtumCorePass> {
public:
  void runOnOperation() override {
    SmallVector<ondrix::ondsp::CxButterflyOp> candidates;
    getOperation()->walk([&](ondrix::ondsp::CxButterflyOp op) { candidates.push_back(op); });
    for (ondrix::ondsp::CxButterflyOp op : candidates)
      rewriteButterfly(op);
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createConvertOndspCxButterflyToOrtumCorePass() {
  return std::make_unique<ConvertOndspCxButterflyToOrtumCorePass>();
}

#include "ondrix/Conversion/OndrixToOndsp/OndrixToOndsp.h"

#include "ondrix/Conversion/OndspToOrtumCore/OndspToOrtumCore.h"
#include "ondrix/Dialect/ondrix/IR/OndrixDialect.h"
#include "ondrix/Dialect/ondrix/IR/OndrixOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <optional>

using namespace mlir;

namespace {

class FirOpLowering final : public OpConversionPattern<ondrix::ir::FirOp> {
public:
  using OpConversionPattern<ondrix::ir::FirOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ir::FirOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto replacement = rewriter.create<ondrix::ondsp::ReduceMacOp>(
        op.getLoc(), op.getResult().getType(), adaptor.getInput(), adaptor.getCoeffs(),
        op.getNumeric(), op.getProduct().value_or(ondrix::ondsp::ProductAttr()));
    replacement->setAttrs(op->getAttrs());
    rewriter.replaceOp(op, replacement);
    return success();
  }
};

class DotOpLowering final : public OpConversionPattern<ondrix::ir::DotOp> {
public:
  using OpConversionPattern<ondrix::ir::DotOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ir::DotOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto replacement = rewriter.create<ondrix::ondsp::ReduceMacOp>(
        op.getLoc(), op.getResult().getType(), adaptor.getLhs(), adaptor.getRhs(), op.getNumeric(),
        op.getProduct().value_or(ondrix::ondsp::ProductAttr()));
    replacement->setAttrs(op->getAttrs());
    rewriter.replaceOp(op, replacement);
    return success();
  }
};

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
        adaptor.getTwiddle(), layout, op.getNumeric(),
        op.getProduct().value_or(ondrix::ondsp::ProductAttr()),
        op.getScale().value_or(ondrix::ondsp::ScaleAttr()), op.getTrivialTwiddle());
    replacement->setAttrs(op->getAttrs());
    rewriter.replaceOp(op, replacement);
    return success();
  }
};

class QuantizeOpLowering final : public OpConversionPattern<ondrix::ir::QuantizeOp> {
public:
  using OpConversionPattern<ondrix::ir::QuantizeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ir::QuantizeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto replacement = rewriter.create<ondrix::ondsp::ConvertOp>(
        op.getLoc(), op.getResult().getType(), adaptor.getInput(), op.getSrc(), op.getDst());
    replacement->setAttrs(op->getAttrs());
    rewriter.replaceOp(op, replacement);
    return success();
  }
};

class ConvertOndrixToOndspPass
    : public PassWrapper<ConvertOndrixToOndspPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertOndrixToOndspPass)

  StringRef getArgument() const final { return "convert-ondrix-to-ondsp"; }
  StringRef getDescription() const final {
    return "Lower ondrix algorithm intent ops to ondsp semantic ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ondrix::ir::OndrixDialect, ondrix::ondsp::OndspDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<FirOpLowering, DotOpLowering, ButterflyOpLowering, QuantizeOpLowering>(
        &getContext());

    ConversionTarget target(getContext());
    target.addLegalDialect<ondrix::ondsp::OndspDialect>();
    target.addIllegalDialect<ondrix::ir::OndrixDialect>();

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createConvertOndrixToOndspPass() {
  return std::make_unique<ConvertOndrixToOndspPass>();
}

void ondrix::registerConvertOndrixToOndspPass() { PassRegistration<ConvertOndrixToOndspPass>(); }

void ondrix::registerConversionPasses() {
  registerConvertOndrixToOndspPass();
  registerConvertOndspToOrtumCorePass();
}

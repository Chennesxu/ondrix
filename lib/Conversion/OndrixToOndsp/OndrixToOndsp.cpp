#include "ondrix/Conversion/OndrixToOndsp/OndrixToOndsp.h"
#include "OndrixToOndspCommon.h"

#include "ondrix/Dialect/ondrix/IR/OndrixDialect.h"
#include "ondrix/Dialect/ondrix/IR/OndrixOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace ondrix {
#define GEN_PASS_DEF_CONVERTONDRIXTOONDSP
#include "ondrix/Conversion/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

class ConvertOndrixToOndspPass final
    : public ondrix::impl::ConvertOndrixToOndspBase<ConvertOndrixToOndspPass> {
public:
  using ondrix::impl::ConvertOndrixToOndspBase<ConvertOndrixToOndspPass>::ConvertOndrixToOndspBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    RewritePatternSet patterns(&getContext());
    if (vectorizeStaticCfft && fftLoops) {
      module.emitError("vectorize-static-cfft and fft-loops are mutually exclusive alternative "
                       "FFT lowerings; select at most one");
      return signalPassFailure();
    }
    ondrix::conversion::populateOndrixFirFamilyLoweringPatterns(patterns, slidingWindowReuse);
    ondrix::conversion::populateOndrixStatefulLoweringPatterns(patterns);
    ondrix::conversion::populateOndrixSpectralLoweringPatterns(patterns, vectorizeStaticCfft,
                                                               fftLoops);
    ondrix::conversion::populateOndrixElementwiseLoweringPatterns(patterns);
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, cf::ControlFlowDialect, math::MathDialect,
                           scf::SCFDialect, tensor::TensorDialect, vector::VectorDialect,
                           ondrix::ondsp::OndspDialect>();
    target.addIllegalDialect<ondrix::ir::OndrixDialect>();
    // In the canonical pipeline the operations whose reductions have a direct
    // bufferization stay in contract form through this pass: bufferization
    // lowers them to the reduce_mac loops the schedule stage authorizes over,
    // which the scalar tensor lowering here would preempt.
    if (preserveBufferizableReductions) {
      target.addLegalOp<ondrix::ir::FirFilterOp, ondrix::ir::FirDecimateOp, ondrix::ir::Conv1DOp,
                        ondrix::ir::MatmulOp, ondrix::ir::RmsOp>();
      // The f32 profile keeps the unrolled tensor lowering: a per-row table and
      // reduction was measured 1.7x slower at N=64 because the rows stop being
      // independent chains and the coefficients stop being immediates. A
      // preserved f32 DCT needs the row axis batched first.
      target.addDynamicallyLegalOp<ondrix::ir::DctOp>(
          [](ondrix::ir::DctOp op) { return !isa<ondrix::ondsp::FpAttr>(op.getInputNumeric()); });
      // The f32 moving average bufferizes to the windowed-sum loop the
      // schedule stage batches; the fixed profile keeps its tensor lowering.
      target.addDynamicallyLegalOp<ondrix::ir::MovingAverageOp>([](ondrix::ir::MovingAverageOp op) {
        return isa<ondrix::ondsp::FpAttr>(op.getNumeric());
      });
    }

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      return signalPassFailure();
    ondrix::ondsp::summarizeFastPermissions(module);
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createConvertOndrixToOndspPass() {
  return std::make_unique<ConvertOndrixToOndspPass>();
}

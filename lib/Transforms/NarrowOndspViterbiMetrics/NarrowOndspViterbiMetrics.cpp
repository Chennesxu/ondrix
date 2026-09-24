#include "ondrix/Transforms/Passes.h"

#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Support/ViterbiDecoding.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace ondrix {
#define GEN_PASS_DEF_NARROWONDSPVITERBIMETRICS
#include "ondrix/Transforms/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

struct NarrowOndspViterbiMetrics final
    : ondrix::impl::NarrowOndspViterbiMetricsBase<NarrowOndspViterbiMetrics> {
  void runOnOperation() override {
    if (metricBits < 2 || metricBits > 32) {
      getOperation().emitError("metric-bits must lie in [2, 32]");
      return signalPassFailure();
    }
    getOperation()->walk([&](ondrix::ondsp::ViterbiDecodeOp op) {
      auto declared = op.getSymbols().getDefiningOp<ondrix::ondsp::AssumeMagnitudeBoundOp>();
      if (!declared || metricBits >= static_cast<int64_t>(op.getMetricBits()))
        return;
      int64_t rate = static_cast<int64_t>(op.getPolynomials().size());
      int64_t stages = op.getSymbols().getType().getDimSize(0) / rate;
      std::optional<ondrix::ViterbiMetricRealization> chosen = ondrix::chooseViterbiRealization(
          op.getConstraintLength(), rate, stages, metricBits, declared.getBound());
      if (!chosen)
        return;
      op.setMetricBits(chosen->metricBits);
      op.setSymbolBound(chosen->symbolBound);
      op.setUnreachableMetric(chosen->unreachableMetric);
      op.setRenormalizationPeriod(chosen->renormalizationPeriod);
    });
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createNarrowOndspViterbiMetricsPass() {
  return std::make_unique<NarrowOndspViterbiMetrics>();
}

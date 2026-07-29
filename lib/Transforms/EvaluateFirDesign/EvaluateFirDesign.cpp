#include "ondrix/Transforms/Passes.h"

#include "ondrix/Dialect/ondrix/IR/OndrixDialect.h"
#include "ondrix/Dialect/ondrix/IR/OndrixOps.h"
#include "ondrix/Support/GuardedQ15Quantization.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <cmath>
#include <cstdint>

namespace ondrix {
#define GEN_PASS_DEF_EVALUATEONDRIXFIRDESIGN
#include "ondrix/Transforms/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

// Nearest binary64 approximations; the tie guard below covers their error.
constexpr double kPi = 3.14159265358979323846264338327950288;
constexpr double kTwoPi = 6.28318530717958647692528676655900577;

struct QuantizedTable {
  llvm::SmallVector<int16_t> values;
  int64_t saturated = 0;
};

// One round-half-even signed Q1.15 quantization per composite real value,
// through the shared guarded quantizer. The guard rejects any coefficient
// whose binary64 estimate cannot prove the real-valued rounding decision, so
// under the declared evaluation error budget (libm sin/cos, the binary64 pi
// constant, and ratio arithmetic; documented at more than three orders of
// magnitude below the guard) an emitted table equals the quantization of the
// real-valued definition and inherits its exact symmetry.
FailureOr<QuantizedTable> quantizeSignedQ15(Operation *op, llvm::ArrayRef<double> reals) {
  QuantizedTable table;
  table.values.reserve(reals.size());
  for (size_t index = 0; index < reals.size(); ++index) {
    std::optional<ondrix::GuardedQ15Value> quantized = ondrix::quantizeGuardedQ15(reals[index]);
    if (!quantized)
      return op->emitOpError() << "coefficient " << index
                               << " lies inside the 2^-20 quantization tie guard; "
                                  "the design profile fails closed";
    if (quantized->saturated)
      ++table.saturated;
    table.values.push_back(quantized->value);
  }
  for (size_t index = 0, extent = table.values.size(); index < extent; ++index)
    if (table.values[index] != table.values[extent - 1 - index])
      return op->emitOpError("generated design table violates the proven symmetry contract");
  return table;
}

double hammingReal(int64_t n, int64_t extent) {
  return 0.54 - 0.46 * std::cos(kTwoPi * static_cast<double>(n) / static_cast<double>(extent - 1));
}

double hannReal(int64_t n, int64_t extent) {
  return 0.5 - 0.5 * std::cos(kTwoPi * static_cast<double>(n) / static_cast<double>(extent - 1));
}

double blackmanReal(int64_t n, int64_t extent) {
  double phase = kTwoPi * static_cast<double>(n) / static_cast<double>(extent - 1);
  return 0.42 - 0.5 * std::cos(phase) + 0.08 * std::cos(2.0 * phase);
}

// Zeroth-order modified Bessel function of the first kind by its power
// series: I0(x) = sum_k ((x/2)^k / k!)^2. Terms fall off factorially, so
// the truncation error is far below one term at the stopping threshold;
// together with binary64 evaluation error the total stays orders of
// magnitude under the 2^-20 tie guard for beta <= 50 (I0(50) ~ 3e20 is
// well inside binary64 range). The guard remains the fail-closed
// backstop for the whole evaluation chain.
double besselI0Real(double x) {
  double term = 1.0;
  double sum = 1.0;
  for (int k = 1; k < 1000; ++k) {
    double factor = x / (2.0 * static_cast<double>(k));
    term *= factor * factor;
    sum += term;
    if (term < sum * 1e-30)
      break;
  }
  return sum;
}

double kaiserReal(int64_t n, int64_t extent, int64_t betaNum, int64_t betaDen) {
  double beta = static_cast<double>(betaNum) / static_cast<double>(betaDen);
  double center = static_cast<double>(extent - 1) / 2.0;
  double ratio = (static_cast<double>(n) - center) / center;
  return besselI0Real(beta * std::sqrt(1.0 - ratio * ratio)) / besselI0Real(beta);
}

double sincReal(double x) {
  if (x == 0.0)
    return 1.0;
  double scaled = kPi * x;
  return std::sin(scaled) / scaled;
}

double lowpassReal(int64_t n, int64_t extent, int64_t cutoffNum, int64_t cutoffDen) {
  int64_t center = (extent - 1) / 2;
  double doubledCutoff = (2.0 * static_cast<double>(cutoffNum)) / static_cast<double>(cutoffDen);
  return doubledCutoff * sincReal(doubledCutoff * static_cast<double>(n - center)) *
         hammingReal(n, extent);
}

LogicalResult replaceWithConstant(Operation *op, RankedTensorType type,
                                  llvm::ArrayRef<double> reals, NamedAttrList provenance) {
  FailureOr<QuantizedTable> table = quantizeSignedQ15(op, reals);
  if (failed(table))
    return failure();
  OpBuilder builder(op);
  auto elements = DenseElementsAttr::get(type, llvm::ArrayRef<int16_t>(table->values));
  auto constant = builder.create<arith::ConstantOp>(op->getLoc(), elements);
  provenance.append("saturated", builder.getI64IntegerAttr(table->saturated));
  constant->setAttr("ondrix.design_provenance", provenance.getDictionary(builder.getContext()));
  op->getResult(0).replaceAllUsesWith(constant.getResult());
  op->erase();
  return success();
}

template <typename WindowOp>
LogicalResult evaluateWindow(WindowOp op, double (*windowReal)(int64_t, int64_t), StringRef kind) {
  RankedTensorType type = op.getCoefficients().getType();
  int64_t extent = type.getDimSize(0);
  llvm::SmallVector<double> reals;
  reals.reserve(extent);
  for (int64_t n = 0; n < extent; ++n)
    reals.push_back(windowReal(n, extent));
  NamedAttrList provenance;
  provenance.append("kind", StringAttr::get(op.getContext(), kind));
  return replaceWithConstant(op, type, reals, std::move(provenance));
}

LogicalResult evaluateKaiser(ondrix::ir::WindowKaiserOp op) {
  RankedTensorType type = op.getCoefficients().getType();
  int64_t extent = type.getDimSize(0);
  llvm::SmallVector<double> reals;
  reals.reserve(extent);
  for (int64_t n = 0; n < extent; ++n)
    reals.push_back(kaiserReal(n, extent, op.getBetaNum(), op.getBetaDen()));
  NamedAttrList provenance;
  provenance.append("kind", StringAttr::get(op.getContext(), "window_kaiser"));
  provenance.append("beta_num",
                    IntegerAttr::get(IntegerType::get(op.getContext(), 64), op.getBetaNum()));
  provenance.append("beta_den",
                    IntegerAttr::get(IntegerType::get(op.getContext(), 64), op.getBetaDen()));
  return replaceWithConstant(op, type, reals, std::move(provenance));
}

LogicalResult evaluateDesign(ondrix::ir::FirDesignWindowedSincOp op) {
  RankedTensorType type = op.getCoefficients().getType();
  int64_t extent = type.getDimSize(0);
  int64_t center = (extent - 1) / 2;
  bool highpass = op.getResponse() == ondrix::ir::FirDesignResponse::Highpass;
  llvm::SmallVector<double> reals;
  reals.reserve(extent);
  for (int64_t n = 0; n < extent; ++n) {
    double lowpass = lowpassReal(n, extent, op.getCutoffNum(), op.getCutoffDen());
    reals.push_back(highpass ? (n == center ? 1.0 : 0.0) - lowpass : lowpass);
  }
  NamedAttrList provenance;
  provenance.append("kind", StringAttr::get(op.getContext(), "fir_design_windowed_sinc"));
  provenance.append(
      "response",
      StringAttr::get(op.getContext(), ondrix::ir::stringifyFirDesignResponse(op.getResponse())));
  provenance.append("cutoff_num",
                    IntegerAttr::get(IntegerType::get(op.getContext(), 64), op.getCutoffNum()));
  provenance.append("cutoff_den",
                    IntegerAttr::get(IntegerType::get(op.getContext(), 64), op.getCutoffDen()));
  return replaceWithConstant(op, type, reals, std::move(provenance));
}

class EvaluateOndrixFirDesignPass final
    : public ondrix::impl::EvaluateOndrixFirDesignBase<EvaluateOndrixFirDesignPass> {
public:
  using ondrix::impl::EvaluateOndrixFirDesignBase<
      EvaluateOndrixFirDesignPass>::EvaluateOndrixFirDesignBase;

  void runOnOperation() override {
    llvm::SmallVector<Operation *> designs;
    getOperation().walk([&](Operation *op) {
      if (isa<ondrix::ir::WindowHammingOp, ondrix::ir::WindowHannOp, ondrix::ir::WindowBlackmanOp,
              ondrix::ir::WindowKaiserOp, ondrix::ir::FirDesignWindowedSincOp>(op))
        designs.push_back(op);
    });
    for (Operation *op : designs) {
      LogicalResult result =
          llvm::TypeSwitch<Operation *, LogicalResult>(op)
              .Case<ondrix::ir::WindowHammingOp>(
                  [](auto window) { return evaluateWindow(window, hammingReal, "window_hamming"); })
              .Case<ondrix::ir::WindowHannOp>(
                  [](auto window) { return evaluateWindow(window, hannReal, "window_hann"); })
              .Case<ondrix::ir::WindowBlackmanOp>([](auto window) {
                return evaluateWindow(window, blackmanReal, "window_blackman");
              })
              .Case<ondrix::ir::WindowKaiserOp>([](auto window) { return evaluateKaiser(window); })
              .Case<ondrix::ir::FirDesignWindowedSincOp>(
                  [](auto design) { return evaluateDesign(design); })
              .Default([](Operation *) { return failure(); });
      if (failed(result))
        return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createEvaluateOndrixFirDesignPass() {
  return std::make_unique<EvaluateOndrixFirDesignPass>();
}

#include "ondrix/Transforms/Passes.h"

#include "ondrix/Dialect/ondrix/IR/OndrixDialect.h"
#include "ondrix/Dialect/ondrix/IR/OndrixOps.h"

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

// Admissibility distance from a Q15 rounding tie, in LSB units (2^-20).
constexpr double kTieGuardLsb = 9.5367431640625e-07;

struct QuantizedTable {
  llvm::SmallVector<int16_t> values;
  int64_t saturated = 0;
};

// One round-half-even signed Q1.15 quantization per composite real value.
// The guard rejects any coefficient whose binary64 estimate cannot prove the
// real-valued rounding decision, so an emitted table is independent of host
// libm rounding and inherits the exact real symmetry.
FailureOr<QuantizedTable> quantizeSignedQ15(Operation *op, llvm::ArrayRef<double> reals) {
  QuantizedTable table;
  table.values.reserve(reals.size());
  for (size_t index = 0; index < reals.size(); ++index) {
    double scaled = reals[index] * 32768.0;
    double lower = std::floor(scaled);
    double fraction = scaled - lower;
    if (std::fabs(fraction - 0.5) < kTieGuardLsb)
      return op->emitOpError() << "coefficient " << index
                               << " lies inside the 2^-20 quantization tie guard; "
                                  "the design profile fails closed";
    int64_t quantized = static_cast<int64_t>(lower) + (fraction > 0.5 ? 1 : 0);
    if (quantized > 32767) {
      quantized = 32767;
      ++table.saturated;
    } else if (quantized < -32768) {
      quantized = -32768;
      ++table.saturated;
    }
    table.values.push_back(static_cast<int16_t>(quantized));
  }
  for (size_t index = 0, extent = table.values.size(); index < extent; ++index)
    if (table.values[index] != table.values[extent - 1 - index])
      return op->emitOpError("generated design table violates the proven symmetry contract");
  return table;
}

double hammingReal(int64_t n, int64_t extent) {
  return 0.54 -
         0.46 * std::cos(kTwoPi * static_cast<double>(n) / static_cast<double>(extent - 1));
}

double sincReal(double x) {
  if (x == 0.0)
    return 1.0;
  double scaled = kPi * x;
  return std::sin(scaled) / scaled;
}

double lowpassReal(int64_t n, int64_t extent, int64_t cutoffNum, int64_t cutoffDen) {
  int64_t center = (extent - 1) / 2;
  double doubledCutoff =
      (2.0 * static_cast<double>(cutoffNum)) / static_cast<double>(cutoffDen);
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
  constant->setAttr("ondrix.design_provenance",
                    provenance.getDictionary(builder.getContext()));
  op->getResult(0).replaceAllUsesWith(constant.getResult());
  op->erase();
  return success();
}

LogicalResult evaluateWindow(ondrix::ir::WindowHammingOp op) {
  RankedTensorType type = op.getCoefficients().getType();
  int64_t extent = type.getDimSize(0);
  llvm::SmallVector<double> reals;
  reals.reserve(extent);
  for (int64_t n = 0; n < extent; ++n)
    reals.push_back(hammingReal(n, extent));
  NamedAttrList provenance;
  provenance.append("kind", StringAttr::get(op.getContext(), "window_hamming"));
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
  provenance.append("response",
                    StringAttr::get(op.getContext(),
                                    ondrix::ir::stringifyFirDesignResponse(op.getResponse())));
  provenance.append("cutoff_num", IntegerAttr::get(IntegerType::get(op.getContext(), 64),
                                                   op.getCutoffNum()));
  provenance.append("cutoff_den", IntegerAttr::get(IntegerType::get(op.getContext(), 64),
                                                   op.getCutoffDen()));
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
      if (isa<ondrix::ir::WindowHammingOp, ondrix::ir::FirDesignWindowedSincOp>(op))
        designs.push_back(op);
    });
    for (Operation *op : designs) {
      LogicalResult result =
          llvm::TypeSwitch<Operation *, LogicalResult>(op)
              .Case<ondrix::ir::WindowHammingOp>([](auto window) { return evaluateWindow(window); })
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

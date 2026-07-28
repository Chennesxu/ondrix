#include "ondrix/Transforms/Passes.h"

#include "ondrix/Dialect/ondrix/IR/OndrixDialect.h"
#include "ondrix/Dialect/ondrix/IR/OndrixOps.h"

#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <cstdint>

namespace ondrix {
#define GEN_PASS_DEF_MERGEONDRIXGAINCASCADES
#include "ondrix/Transforms/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

// The exact ondrix.gain contract for one element: exact integer product,
// round-half-even by 15 in explicit floor-division form, i16 saturation.
int64_t applyGainQ15(int64_t value, int64_t gain) {
  int64_t product = value * gain;
  int64_t quotient = product / 32768;
  int64_t remainder = product % 32768;
  if (remainder < 0) {
    --quotient;
    remainder += 32768;
  }
  if (remainder > 16384 || (remainder == 16384 && (quotient & 1)))
    ++quotient;
  if (quotient > 32767)
    return 32767;
  if (quotient < -32768)
    return -32768;
  return quotient;
}

// The natural merged constant: the round-half-even Q1.15 quantization of
// the exact rational product g1*g2/2^15. This is pure integer arithmetic
// (no binary64, no tie guard needed): applyGainQ15 on the exact integer
// product of the two constants.
int64_t mergedGainQ15(int64_t inner, int64_t outer) { return applyGainQ15(inner, outer); }

// Exhaustive equivalence certificate: the cascade (inner gain first, then
// outer) and the single merged gain must be bit-identical on every one of
// the 65536 possible i16 inputs. There is no sampled or approximate mode —
// either the whole domain agrees or the rewrite does not happen.
bool certifyMerge(int64_t inner, int64_t outer, int64_t merged) {
  for (int64_t value = -32768; value <= 32767; ++value)
    if (applyGainQ15(applyGainQ15(value, inner), outer) != applyGainQ15(value, merged))
      return false;
  return true;
}

class MergeOndrixGainCascadesPass final
    : public ondrix::impl::MergeOndrixGainCascadesBase<MergeOndrixGainCascadesPass> {
public:
  using ondrix::impl::MergeOndrixGainCascadesBase<
      MergeOndrixGainCascadesPass>::MergeOndrixGainCascadesBase;

  void runOnOperation() override {
    // A merge can expose another cascade (three or more gains), so iterate
    // to a fixpoint; each accepted rewrite is certified independently.
    bool changed = true;
    while (changed) {
      changed = false;
      llvm::SmallVector<ondrix::ir::GainOp> outers;
      getOperation().walk([&](ondrix::ir::GainOp op) { outers.push_back(op); });
      for (ondrix::ir::GainOp outer : outers) {
        auto inner = outer.getInput().getDefiningOp<ondrix::ir::GainOp>();
        if (!inner || !inner.getResult().hasOneUse())
          continue;
        // The certificate models exactly one numeric policy; differing
        // policies never merge.
        if (inner.getNumeric() != outer.getNumeric() || inner.getRounding() != outer.getRounding())
          continue;
        int64_t merged = mergedGainQ15(inner.getGain(), outer.getGain());
        if (!certifyMerge(inner.getGain(), outer.getGain(), merged))
          continue;

        OpBuilder builder(outer);
        auto replacement = builder.create<ondrix::ir::GainOp>(
            outer.getLoc(), outer.getResult().getType(), inner.getInput(),
            builder.getI64IntegerAttr(merged), outer.getNumeric(), outer.getRoundingAttr());
        NamedAttrList provenance;
        provenance.append("inner_gain", builder.getI64IntegerAttr(inner.getGain()));
        provenance.append("outer_gain", builder.getI64IntegerAttr(outer.getGain()));
        provenance.append("exhaustive_inputs", builder.getI64IntegerAttr(65536));
        replacement->setAttr("ondrix.gain_merge_provenance",
                             provenance.getDictionary(builder.getContext()));
        outer.getResult().replaceAllUsesWith(replacement.getResult());
        outer.erase();
        inner.erase();
        changed = true;
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createMergeOndrixGainCascadesPass() {
  return std::make_unique<MergeOndrixGainCascadesPass>();
}

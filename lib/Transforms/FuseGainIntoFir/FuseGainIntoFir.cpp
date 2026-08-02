#include "ondrix/Transforms/Passes.h"

#include "ondrix/Dialect/ondrix/IR/OndrixDialect.h"
#include "ondrix/Dialect/ondrix/IR/OndrixOps.h"
#include "ondrix/Support/GainQ15Contract.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

#include <cstdint>
#include <optional>
#include <utility>

namespace ondrix {
#define GEN_PASS_DEF_FUSEONDRIXGAININTOFIR
#include "ondrix/Transforms/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

// The contract arithmetic is shared with the gain-cascade merge; both
// certificates must judge the same `ondrix.gain`.
using ondrix::applyGainQ15;
using ondrix::isAdmittedGainRounding;
using ondrix::quantizeQ15Product;

// The certificate domain: every representable i16 sample the filter can read.
constexpr int64_t kExhaustiveInputs = 65536;

// One refutation of a proposed fused tap: the input that separates the two
// programs and the two exact products that enter the accumulator there.
struct TapWitness {
  int64_t input;
  int64_t gainTerm;
  int64_t fusedTerm;
};

// Exhaustive per-tap equivalence certificate. The unfused program feeds the
// accumulator `applyGainQ15(x, g, rule) * h`; the fused program feeds
// `x * h'`. Both are exact integers and must agree on every one of the 65536
// possible i16 inputs — one uncertified tap refuses the whole filter.
// Term-level equality makes the rewrite independent of everything downstream
// (the pass description carries that argument).
std::optional<TapWitness> certifyTap(int64_t gain, int64_t tap, int64_t fusedTap,
                                     ondrix::ondsp::RoundingMode mode) {
  for (int64_t value = -32768; value <= 32767; ++value) {
    int64_t gainTerm = applyGainQ15(value, gain, mode) * tap;
    int64_t fusedTerm = value * fusedTap;
    if (gainTerm != fusedTerm)
      return TapWitness{value, gainTerm, fusedTerm};
  }
  return std::nullopt;
}

// The second proposal: `g * h / 2^15` when that division is exact and lands
// inside the Q1.15 range, so the fused tap is the scaled tap with no rounding
// of its own. It coincides with the quantized proposal whenever both exist,
// but it is stated separately because a rounding-free candidate is the one a
// reader expects the certificate to admit.
std::optional<int64_t> exactFusedTap(int64_t gain, int64_t tap) {
  int64_t product = gain * tap;
  if (product % 32768 != 0)
    return std::nullopt;
  int64_t quotient = product / 32768;
  if (quotient < -32768 || quotient > 32767)
    return std::nullopt;
  return quotient;
}

// The certified fused tap sequence for a whole filter, or the first tap that
// refutes the rewrite. Repeated (tap, proposal) pairs are certified once; the
// domain sweep is unchanged, only re-executing it is skipped.
struct FilterCertificate {
  llvm::SmallVector<int64_t> fusedTaps;
  int64_t failedIndex = -1;
  int64_t failedTap = 0;
  int64_t failedProposal = 0;
  TapWitness failedWitness{};

  bool certified() const { return failedIndex < 0; }
};

FilterCertificate certifyFilter(llvm::ArrayRef<int64_t> taps, int64_t gain,
                                ondrix::ondsp::RoundingMode mode) {
  FilterCertificate certificate;
  llvm::DenseMap<std::pair<int64_t, int64_t>, bool> cache;
  for (auto [index, tap] : llvm::enumerate(taps)) {
    int64_t quantized = quantizeQ15Product(gain, tap, mode);
    llvm::SmallVector<int64_t, 2> proposals{quantized};
    if (std::optional<int64_t> exact = exactFusedTap(gain, tap); exact && *exact != quantized)
      proposals.push_back(*exact);

    std::optional<int64_t> accepted;
    for (int64_t proposal : proposals) {
      auto [entry, inserted] = cache.try_emplace({tap, proposal}, false);
      if (inserted)
        entry->second = !certifyTap(gain, tap, proposal, mode).has_value();
      if (entry->second) {
        accepted = proposal;
        break;
      }
    }
    if (!accepted) {
      // The reported refutation is always the one against the natural
      // quantized proposal, so the record names the rewrite a reader would
      // have expected to fire.
      certificate.failedIndex = static_cast<int64_t>(index);
      certificate.failedTap = tap;
      certificate.failedProposal = quantized;
      certificate.failedWitness = *certifyTap(gain, tap, quantized, mode);
      certificate.fusedTaps.clear();
      return certificate;
    }
    certificate.fusedTaps.push_back(*accepted);
  }
  return certificate;
}

class FuseOndrixGainIntoFirPass final
    : public ondrix::impl::FuseOndrixGainIntoFirBase<FuseOndrixGainIntoFirPass> {
public:
  using ondrix::impl::FuseOndrixGainIntoFirBase<
      FuseOndrixGainIntoFirPass>::FuseOndrixGainIntoFirBase;

  void runOnOperation() override {
    // A fusion exposes the fused filter's new input, which may itself be a
    // gain, so iterate to a fixpoint; each accepted rewrite is certified
    // independently.
    bool changed = true;
    while (changed) {
      changed = false;
      llvm::SmallVector<ondrix::ir::FirFilterOp> filters;
      getOperation().walk([&](ondrix::ir::FirFilterOp op) { filters.push_back(op); });
      for (ondrix::ir::FirFilterOp filter : filters)
        changed |= tryFuse(filter);
    }
  }

private:
  bool tryFuse(ondrix::ir::FirFilterOp filter) {
    auto gain = filter.getInput().getDefiningOp<ondrix::ir::GainOp>();
    if (!gain)
      return false;
    // A gain observed anywhere else is not the filter's private boundary;
    // deleting it would change the other consumers.
    if (!gain.getResult().hasOneUse())
      return false;
    // One numeric policy, and it is the Q1.15 policy the gain verifier pins.
    if (filter.getNumeric() != gain.getNumeric())
      return false;
    ondrix::ondsp::RoundingMode mode = gain.getRounding();
    if (!isAdmittedGainRounding(mode))
      return false;

    DenseIntElementsAttr coefficients;
    if (!matchPattern(filter.getCoeffs(), m_Constant(&coefficients)))
      return false;
    auto coefficientType = dyn_cast<RankedTensorType>(coefficients.getType());
    if (!coefficientType || !coefficientType.hasStaticShape() ||
        !coefficientType.getElementType().isSignlessInteger(16) ||
        coefficientType.getNumElements() == 0)
      return false;

    llvm::SmallVector<int64_t> taps;
    taps.reserve(coefficientType.getNumElements());
    for (const llvm::APInt &tap : coefficients.getValues<llvm::APInt>())
      taps.push_back(tap.getSExtValue());

    FilterCertificate certificate = certifyFilter(taps, gain.getGain(), mode);
    if (!certificate.certified()) {
      if (recordRefusals)
        recordRefusal(filter, gain, mode, certificate);
      return false;
    }

    OpBuilder builder(filter);
    llvm::SmallVector<llvm::APInt> fused;
    fused.reserve(certificate.fusedTaps.size());
    for (int64_t tap : certificate.fusedTaps)
      fused.emplace_back(16, static_cast<uint64_t>(tap), /*isSigned=*/true);
    Value fusedCoefficients = builder.create<arith::ConstantOp>(
        filter.getLoc(), DenseIntElementsAttr::get(coefficientType, fused));

    // Cloning keeps every declared policy of the original filter — boundary,
    // accumulator, product, export — because the certificate proves the
    // rewrite cannot interact with any of them.
    Operation *fusedFilter = filter->clone();
    fusedFilter->setOperand(0, gain.getInput());
    fusedFilter->setOperand(1, fusedCoefficients);
    NamedAttrList provenance;
    provenance.append("gain", builder.getI64IntegerAttr(gain.getGain()));
    // The certificate is only valid under the tie rule it was evaluated with,
    // so the provenance records which one that was.
    provenance.append("rounding",
                      builder.getStringAttr(ondrix::ondsp::stringifyRoundingMode(mode)));
    provenance.append("exhaustive_inputs", builder.getI64IntegerAttr(kExhaustiveInputs));
    provenance.append("certified_taps",
                      builder.getI64IntegerAttr(static_cast<int64_t>(taps.size())));
    fusedFilter->setAttr("ondrix.gain_fusion_provenance",
                         provenance.getDictionary(builder.getContext()));
    builder.insert(fusedFilter);

    Value staleCoefficients = filter.getCoeffs();
    filter.getResult().replaceAllUsesWith(fusedFilter->getResult(0));
    filter.erase();
    gain.erase();
    // The original table is pure and may now be unreferenced; leaving it
    // behind would misrepresent the rewritten program.
    if (staleCoefficients.use_empty())
      if (Operation *definition = staleCoefficients.getDefiningOp())
        if (isa<arith::ConstantOp>(definition))
          definition->erase();
    return true;
  }

  // Diagnostic mode only: the refused filter keeps its meaning and its
  // operands, and gains a record of why the certificate rejected it.
  void recordRefusal(ondrix::ir::FirFilterOp filter, ondrix::ir::GainOp gain,
                     ondrix::ondsp::RoundingMode mode, const FilterCertificate &certificate) {
    Builder builder(filter.getContext());
    NamedAttrList refusal;
    refusal.append("gain", builder.getI64IntegerAttr(gain.getGain()));
    refusal.append("rounding", builder.getStringAttr(ondrix::ondsp::stringifyRoundingMode(mode)));
    refusal.append("tap_index", builder.getI64IntegerAttr(certificate.failedIndex));
    refusal.append("tap", builder.getI64IntegerAttr(certificate.failedTap));
    refusal.append("proposed_tap", builder.getI64IntegerAttr(certificate.failedProposal));
    refusal.append("witness_input", builder.getI64IntegerAttr(certificate.failedWitness.input));
    refusal.append("gain_term", builder.getI64IntegerAttr(certificate.failedWitness.gainTerm));
    refusal.append("fused_term", builder.getI64IntegerAttr(certificate.failedWitness.fusedTerm));
    filter->setAttr("ondrix.gain_fusion_refusal", refusal.getDictionary(filter.getContext()));
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createFuseOndrixGainIntoFirPass() {
  return std::make_unique<FuseOndrixGainIntoFirPass>();
}

std::unique_ptr<Pass>
ondrix::createFuseOndrixGainIntoFirPass(const FuseOndrixGainIntoFirOptions &options) {
  return std::make_unique<FuseOndrixGainIntoFirPass>(options);
}

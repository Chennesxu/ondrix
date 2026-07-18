#ifndef ONDRIX_ANALYSIS_FIXEDPOINTPREFIXRANGEANALYSIS_H
#define ONDRIX_ANALYSIS_FIXEDPOINTPREFIXRANGEANALYSIS_H

#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/Support/LLVM.h"

#include <cstdint>
#include <utility>

namespace ondrix::analysis {

/// Conservative closed interval for a pre-overflow fixed-point raw value. Both
/// endpoints use the same bit width and `frac` domain.
struct FixedPointRawInterval {
  llvm::APInt lower;
  llvm::APInt upper;
  unsigned frac;
};

/// Computes the exact raw interval of a signed full-width product between any
/// value in `numeric`'s storage domain and one constant raw coefficient.
mlir::FailureOr<FixedPointRawInterval>
computeSignedFullProductInterval(ondsp::FixedAttr numeric, const llvm::APInt &coefficient);

/// Adds two independent raw intervals in a widened signed domain. Both inputs
/// must use the same fractional position.
mlir::FailureOr<FixedPointRawInterval> addFixedPointRawIntervals(const FixedPointRawInterval &lhs,
                                                                 const FixedPointRawInterval &rhs);

/// Schedule indices and raw coefficient values for one proposed distributive
/// pair. Original indices address the complete source update schedule;
/// `reassociatedIndex` addresses the complete candidate schedule.
struct CoefficientPair {
  int64_t originalLhsIndex;
  int64_t originalRhsIndex;
  int64_t reassociatedIndex;
  llvm::APInt lhsCoefficient;
  llvm::APInt rhsCoefficient;
};

/// Maps an unchanged source update into the complete candidate schedule.
struct PassthroughUpdate {
  int64_t originalIndex;
  int64_t reassociatedIndex;
};

class FixedPointPrefixRangePlanner;

/// Opaque evidence produced only after the prefix-range planner validates a
/// complete distributive-pairing plan. Callers may inspect but cannot create
/// evidence that authorizes a rewrite.
class DistributivePairingEvidence final {
public:
  bool isExactModulo() const { return kind == Kind::ExactModulo; }
  bool isProvenNoOverflow() const { return kind == Kind::ProvenNoOverflow; }

private:
  friend class FixedPointPrefixRangePlanner;

  enum class Kind {
    ExactModulo,
    ProvenNoOverflow,
  };

  explicit DistributivePairingEvidence(Kind kind) : kind(kind) {}

  Kind kind;
};

using DistributivePairingConsumer = llvm::function_ref<mlir::LogicalResult(
    const ondsp::DistributivePairingSemantics &, const DistributivePairingEvidence &)>;

/// A move-only decision bound to the operation, numeric domain, complete
/// schedule mapping, initial range, and update intervals analyzed. Legality is
/// exposed only to a one-shot callback after fresh facts have been validated.
class DistributivePairingPlan final {
public:
  DistributivePairingPlan(const DistributivePairingPlan &) = delete;
  DistributivePairingPlan &operator=(const DistributivePairingPlan &) = delete;
  DistributivePairingPlan(DistributivePairingPlan &&other);
  DistributivePairingPlan &operator=(DistributivePairingPlan &&other);

  mlir::LogicalResult consumeIfValid(mlir::Operation *operation, ondsp::FixedAttr numeric,
                                     ondsp::ProductAttr product, ondsp::AccType accumulator,
                                     llvm::ArrayRef<CoefficientPair> coefficientPairs,
                                     llvm::ArrayRef<PassthroughUpdate> passthroughUpdates,
                                     const FixedPointRawInterval &initial,
                                     llvm::ArrayRef<FixedPointRawInterval> originalUpdates,
                                     llvm::ArrayRef<FixedPointRawInterval> reassociatedUpdates,
                                     DistributivePairingConsumer consumer) &&;

private:
  friend class FixedPointPrefixRangePlanner;
  DistributivePairingPlan(mlir::Operation *subject, ondsp::DistributivePairingSemantics semantics,
                          DistributivePairingEvidence evidence, ondsp::FixedAttr numeric,
                          ondsp::ProductAttr product, ondsp::AccType accumulator,
                          llvm::ArrayRef<CoefficientPair> coefficientPairs,
                          llvm::ArrayRef<PassthroughUpdate> passthroughUpdates,
                          const FixedPointRawInterval &initial,
                          llvm::ArrayRef<FixedPointRawInterval> originalUpdates,
                          llvm::ArrayRef<FixedPointRawInterval> reassociatedUpdates)
      : subject(subject), semantics(std::move(semantics)), evidence(evidence), numeric(numeric),
        product(product), accumulator(accumulator), coefficientPairs(coefficientPairs),
        passthroughUpdates(passthroughUpdates), initial(initial), originalUpdates(originalUpdates),
        reassociatedUpdates(reassociatedUpdates) {}

  mlir::Operation *subject;
  ondsp::DistributivePairingSemantics semantics;
  DistributivePairingEvidence evidence;
  ondsp::FixedAttr numeric;
  ondsp::ProductAttr product;
  ondsp::AccType accumulator;
  llvm::SmallVector<CoefficientPair> coefficientPairs;
  llvm::SmallVector<PassthroughUpdate> passthroughUpdates;
  FixedPointRawInterval initial;
  llvm::SmallVector<FixedPointRawInterval> originalUpdates;
  llvm::SmallVector<FixedPointRawInterval> reassociatedUpdates;
  bool consumed = false;
};

/// Stateless planner for prefix containment and transform-specific legality.
class FixedPointPrefixRangePlanner final {
public:
  static mlir::LogicalResult
  proveAllPrefixesFit(unsigned accumulatorWidth, const FixedPointRawInterval &initial,
                      llvm::ArrayRef<FixedPointRawInterval> originalUpdates,
                      llvm::ArrayRef<FixedPointRawInterval> reassociatedUpdates);

  /// Builds a plan for complete source and candidate update schedules. Every
  /// source and candidate index must occur exactly once in either a pair or a
  /// passthrough mapping.
  static mlir::FailureOr<DistributivePairingPlan> planDistributivePairing(
      mlir::Operation *subject, ondsp::FixedAttr numeric, ondsp::ProductAttr product,
      ondsp::AccType accumulator, llvm::ArrayRef<CoefficientPair> coefficientPairs,
      llvm::ArrayRef<PassthroughUpdate> passthroughUpdates, const FixedPointRawInterval &initial,
      llvm::ArrayRef<FixedPointRawInterval> originalUpdates,
      llvm::ArrayRef<FixedPointRawInterval> reassociatedUpdates);
};

} // namespace ondrix::analysis

#endif // ONDRIX_ANALYSIS_FIXEDPOINTPREFIXRANGEANALYSIS_H

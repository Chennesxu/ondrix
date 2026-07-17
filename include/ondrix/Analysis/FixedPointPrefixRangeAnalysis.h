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

using DistributivePairingConsumer = llvm::function_ref<mlir::LogicalResult(
    const ondsp::DistributivePairingSemantics &, const ondsp::TransformLegality &)>;

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
                          ondsp::TransformLegality legality, ondsp::FixedAttr numeric,
                          ondsp::ProductAttr product, ondsp::AccType accumulator,
                          llvm::ArrayRef<CoefficientPair> coefficientPairs,
                          llvm::ArrayRef<PassthroughUpdate> passthroughUpdates,
                          const FixedPointRawInterval &initial,
                          llvm::ArrayRef<FixedPointRawInterval> originalUpdates,
                          llvm::ArrayRef<FixedPointRawInterval> reassociatedUpdates)
      : subject(subject), semantics(std::move(semantics)), legality(legality), numeric(numeric),
        product(product), accumulator(accumulator), coefficientPairs(coefficientPairs),
        passthroughUpdates(passthroughUpdates), initial(initial), originalUpdates(originalUpdates),
        reassociatedUpdates(reassociatedUpdates) {}

  mlir::Operation *subject;
  ondsp::DistributivePairingSemantics semantics;
  ondsp::TransformLegality legality;
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

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

using DistributivePairingConsumer = llvm::function_ref<mlir::LogicalResult(
    const ondsp::DistributivePairingSemantics &, llvm::ArrayRef<llvm::APInt>)>;

/// A move-only decision bound to the operation, numeric domain, and complete
/// coefficient sequence analyzed. Validated semantics and coefficients are
/// exposed only to a one-shot callback after fresh facts have been checked.
class DistributivePairingPlan final {
public:
  DistributivePairingPlan(const DistributivePairingPlan &) = delete;
  DistributivePairingPlan &operator=(const DistributivePairingPlan &) = delete;
  DistributivePairingPlan(DistributivePairingPlan &&other);
  DistributivePairingPlan &operator=(DistributivePairingPlan &&other);

  mlir::LogicalResult consumeIfValid(mlir::Operation *operation, ondsp::FixedAttr numeric,
                                     ondsp::ProductAttr product, ondsp::AccType accumulator,
                                     llvm::ArrayRef<llvm::APInt> coefficients,
                                     DistributivePairingConsumer consumer) &&;

private:
  friend class FixedPointPrefixRangePlanner;
  DistributivePairingPlan(mlir::Operation *subject, ondsp::DistributivePairingSemantics semantics,
                          ondsp::FixedAttr numeric, ondsp::ProductAttr product,
                          ondsp::AccType accumulator, llvm::ArrayRef<llvm::APInt> coefficients)
      : subject(subject), semantics(std::move(semantics)), numeric(numeric), product(product),
        accumulator(accumulator), coefficients(coefficients) {}

  mlir::Operation *subject;
  ondsp::DistributivePairingSemantics semantics;
  ondsp::FixedAttr numeric;
  ondsp::ProductAttr product;
  ondsp::AccType accumulator;
  llvm::SmallVector<llvm::APInt> coefficients;
  bool consumed = false;
};

/// Stateless planner for prefix containment and transform-specific legality.
class FixedPointPrefixRangePlanner final {
public:
  static mlir::LogicalResult
  proveAllPrefixesFit(unsigned accumulatorWidth, const FixedPointRawInterval &initial,
                      llvm::ArrayRef<FixedPointRawInterval> originalUpdates,
                      llvm::ArrayRef<FixedPointRawInterval> reassociatedUpdates);

  /// Builds a zero-seeded symmetric-pairing plan. Product intervals for the
  /// source and candidate schedules are derived internally from the complete
  /// signed input domain and the supplied raw coefficients.
  static mlir::FailureOr<DistributivePairingPlan>
  planZeroSeededSymmetricPairing(mlir::Operation *subject, ondsp::FixedAttr numeric,
                                 ondsp::ProductAttr product, ondsp::AccType accumulator,
                                 llvm::ArrayRef<llvm::APInt> coefficients);
};

} // namespace ondrix::analysis

#endif // ONDRIX_ANALYSIS_FIXEDPOINTPREFIXRANGEANALYSIS_H

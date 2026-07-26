#ifndef ONDRIX_ANALYSIS_FIXEDPOINTPREFIXRANGEANALYSIS_H
#define ONDRIX_ANALYSIS_FIXEDPOINTPREFIXRANGEANALYSIS_H

#include "ondrix/Analysis/ConstantSequenceAnalysis.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/JSON.h"

#include "mlir/Support/LLVM.h"

#include <cstddef>
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

/// Experimental, serializable evidence for one zero-seeded constant chunk
/// reassociation. This is an audit artifact, not a legality authority: every
/// consumer must revalidate it against the current IR and planner result.
struct NoOverflowChunkReassociationTrace {
  static constexpr int64_t schemaVersion = 1;

  int64_t subjectOrdinal = -1;
  unsigned numericStorageWidth = 0;
  unsigned numericFrac = 0;
  unsigned accumulatorStorageWidth = 0;
  unsigned accumulatorFrac = 0;
  unsigned productRawWidth = 0;
  unsigned productFrac = 0;
  int64_t chunkWidth = 0;
  llvm::SmallVector<llvm::APInt> coefficients;
  llvm::SmallVector<FixedPointRawInterval> originalPrefixes;
  llvm::SmallVector<FixedPointRawInterval> reassociatedPrefixes;
};

/// Resource limits applied before materializing untrusted trace fields.
struct NoOverflowChunkReassociationTraceParseLimits {
  size_t maxCoefficients = 65536;
  size_t maxPrefixes = 65537;
  unsigned maxAPIntWidth = 4096;
};

llvm::json::Object toJSON(const NoOverflowChunkReassociationTrace &trace);
mlir::FailureOr<NoOverflowChunkReassociationTrace>
parseNoOverflowChunkReassociationTrace(const llvm::json::Value &value,
                                       NoOverflowChunkReassociationTraceParseLimits limits = {});
mlir::LogicalResult
verifyNoOverflowChunkReassociationTrace(const NoOverflowChunkReassociationTrace &trace);
bool areEquivalent(const NoOverflowChunkReassociationTrace &lhs,
                   const NoOverflowChunkReassociationTrace &rhs);

/// Computes the exact raw interval of a signed full-width product between any
/// value in `numeric`'s storage domain and one constant raw coefficient.
mlir::FailureOr<FixedPointRawInterval>
computeSignedFullProductInterval(ondsp::FixedAttr numeric, const llvm::APInt &coefficient);

/// Adds two independent raw intervals in a widened signed domain. Both inputs
/// must use the same fractional position.
mlir::FailureOr<FixedPointRawInterval> addFixedPointRawIntervals(const FixedPointRawInterval &lhs,
                                                                 const FixedPointRawInterval &rhs);

/// Returns whether every raw value in `interval` fits a signed implementation
/// value of `width` bits.
bool fitsSignedImplementationWidth(const FixedPointRawInterval &interval, unsigned width);

/// Result of analyzing the numeric schedule used by fixed-width horizontal
/// chunk reduction. This classification does not authorize an IR rewrite:
/// callers must separately validate the operation, product policy, zero seed,
/// coefficient provenance, and subject identity.
enum class ConstantChunkReassociationStatus {
  Authorized,
  InvalidInput,
  ImplementationTermOverflow,
  PrefixOverflow,
};

struct ConstantChunkReassociationAnalysis {
  ConstantChunkReassociationStatus status = ConstantChunkReassociationStatus::InvalidInput;
  llvm::SmallVector<FixedPointRawInterval> originalUpdates;
  llvm::SmallVector<FixedPointRawInterval> reassociatedUpdates;
};

/// Classifies one zero-seeded constant chunk schedule. Product intervals are
/// derived from the complete signed input domain. `implementationTermWidth`
/// is the actual integer width used by the horizontal-sum consumer. The caller
/// must separately verify that the accumulator fractional position is
/// `2 * numericFrac`; only its storage width participates in this raw-range
/// classification.
ConstantChunkReassociationAnalysis analyzeZeroSeededConstantChunkReassociation(
    unsigned numericStorageWidth, unsigned numericFrac, unsigned accumulatorWidth,
    llvm::ArrayRef<llvm::APInt> coefficients, int64_t chunkWidth, unsigned implementationTermWidth);

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

using ChunkReassociationConsumer = llvm::function_ref<mlir::LogicalResult(
    const ondsp::ProductSemantics &, llvm::ArrayRef<llvm::APInt>, int64_t,
    const NoOverflowChunkReassociationTrace &)>;

/// A move-only decision authorizing fixed-width chunk reassociation for one
/// zero-seeded constant reduction. The planner owns schedule construction;
/// consumers cannot supply narrower update intervals.
class NoOverflowChunkReassociationPlan final {
public:
  NoOverflowChunkReassociationPlan(const NoOverflowChunkReassociationPlan &) = delete;
  NoOverflowChunkReassociationPlan &operator=(const NoOverflowChunkReassociationPlan &) = delete;
  NoOverflowChunkReassociationPlan(NoOverflowChunkReassociationPlan &&other);
  NoOverflowChunkReassociationPlan &operator=(NoOverflowChunkReassociationPlan &&other);

  mlir::LogicalResult consumeIfValid(ondsp::ReduceMacOp reduction, int64_t chunkWidth,
                                     ChunkReassociationConsumer consumer) &&;

private:
  friend class FixedPointPrefixRangePlanner;
  NoOverflowChunkReassociationPlan(mlir::Operation *subject, mlir::Value coefficientSource,
                                   ondsp::ProductSemantics productSemantics,
                                   ondsp::FixedAttr numeric, ondsp::ProductAttr product,
                                   ondsp::AccType accumulator,
                                   llvm::ArrayRef<llvm::APInt> coefficients, int64_t chunkWidth,
                                   NoOverflowChunkReassociationTrace trace)
      : subject(subject), coefficientSource(coefficientSource), productSemantics(productSemantics),
        numeric(numeric), product(product), accumulator(accumulator), coefficients(coefficients),
        chunkWidth(chunkWidth), trace(std::move(trace)) {}

  mlir::Operation *subject;
  mlir::Value coefficientSource;
  ondsp::ProductSemantics productSemantics;
  ondsp::FixedAttr numeric;
  ondsp::ProductAttr product;
  ondsp::AccType accumulator;
  llvm::SmallVector<llvm::APInt> coefficients;
  int64_t chunkWidth;
  NoOverflowChunkReassociationTrace trace;
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

  /// Builds a plan for replacing ordered constant products with fixed-width
  /// horizontal chunk sums followed by the original scalar tail. Both source
  /// and candidate schedules are derived internally and must fit a saturating
  /// accumulator for the complete signed input domain.
  static mlir::FailureOr<NoOverflowChunkReassociationPlan>
  planZeroSeededConstantChunkReduction(ondsp::ReduceMacOp reduction,
                                       const ondrix::ConstantIntegerMemRefFacts &coefficients,
                                       int64_t chunkWidth);
};

} // namespace ondrix::analysis

#endif // ONDRIX_ANALYSIS_FIXEDPOINTPREFIXRANGEANALYSIS_H

#ifndef ONDRIX_ANALYSIS_CANONICALTWIDDLEANALYSIS_H
#define ONDRIX_ANALYSIS_CANONICALTWIDDLEANALYSIS_H

#include "ondrix/Dialect/ondsp/IR/OndspOps.h"

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"

#include "mlir/Support/LogicalResult.h"

#include <optional>

namespace ondrix::analysis {

enum class CanonicalPackedQ15TwiddleIdentity {
  One,
  MinusJ,
};

enum class CanonicalPackedQ15TwiddleStatus {
  Authorized,
  UnsupportedValueDomain,
  UnsupportedLayout,
  UnsupportedNumeric,
  UnsupportedProduct,
  UnsupportedScale,
  UnsupportedRounding,
  UnsupportedOverflow,
  NonConstantTwiddle,
  NonCanonicalTwiddle,
};

struct CanonicalPackedQ15TwiddleClassification {
  CanonicalPackedQ15TwiddleStatus status = CanonicalPackedQ15TwiddleStatus::UnsupportedValueDomain;
  std::optional<CanonicalPackedQ15TwiddleIdentity> identity;
};

llvm::StringRef stringifyCanonicalPackedQ15TwiddleStatus(CanonicalPackedQ15TwiddleStatus status);

/// Classifies the two exact packed-Q15 twiddle identities supported by the
/// current scalar specialization. This result describes the current operation
/// but does not authorize mutation; use a one-shot plan for transformation.
CanonicalPackedQ15TwiddleClassification
classifyCanonicalPackedQ15Twiddle(ondsp::CxButterflyOp butterfly);

/// Whether the packed-Q15 product carrier may be 32 bits instead of 33 for
/// this butterfly. The wider carrier exists only because the operation admits
/// an ARBITRARY i16 twiddle, whose imaginary cross sum reaches exactly 2^31
/// at `w = (-32768, -32768)`. Both cross sums are bounded by
/// `32768 * (|wr| + |wi|)`, so a constant twiddle with `|wr| + |wi| <= 65535`
/// keeps them inside i32; every quantized unit-modulus twiddle satisfies that
/// with a factor of sqrt(2) to spare, and a runtime twiddle fails closed.
bool packedQ15ProductFitsNarrowCarrier(ondsp::CxButterflyOp butterfly);

using CanonicalPackedQ15TwiddleConsumer =
    llvm::function_ref<mlir::LogicalResult(CanonicalPackedQ15TwiddleIdentity)>;

/// Move-only authorization bound to one butterfly, its twiddle SSA value, and
/// its complete numeric policy. Consumption reclassifies the current operation
/// before exposing the identity to the callback.
class CanonicalPackedQ15TwiddlePlan final {
public:
  CanonicalPackedQ15TwiddlePlan(const CanonicalPackedQ15TwiddlePlan &) = delete;
  CanonicalPackedQ15TwiddlePlan &operator=(const CanonicalPackedQ15TwiddlePlan &) = delete;
  CanonicalPackedQ15TwiddlePlan(CanonicalPackedQ15TwiddlePlan &&other);
  CanonicalPackedQ15TwiddlePlan &operator=(CanonicalPackedQ15TwiddlePlan &&other);

  mlir::LogicalResult consumeIfValid(ondsp::CxButterflyOp butterfly,
                                     CanonicalPackedQ15TwiddleConsumer consumer) &&;

private:
  friend std::optional<CanonicalPackedQ15TwiddlePlan>
  planCanonicalPackedQ15Twiddle(ondsp::CxButterflyOp butterfly);

  CanonicalPackedQ15TwiddlePlan(ondsp::CxButterflyOp butterfly,
                                CanonicalPackedQ15TwiddleIdentity identity);

  mlir::Operation *subject;
  mlir::Value twiddle;
  ondsp::CxLayoutAttr layout;
  mlir::Attribute numeric;
  ondsp::ProductAttr product;
  ondsp::ScaleAttr productScale;
  ondsp::ScaleAttr outputScale;
  CanonicalPackedQ15TwiddleIdentity identity;
  bool consumed = false;
};

std::optional<CanonicalPackedQ15TwiddlePlan>
planCanonicalPackedQ15Twiddle(ondsp::CxButterflyOp butterfly);

} // namespace ondrix::analysis

#endif

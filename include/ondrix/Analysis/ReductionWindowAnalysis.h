#ifndef ONDRIX_ANALYSIS_REDUCTIONWINDOWANALYSIS_H
#define ONDRIX_ANALYSIS_REDUCTIONWINDOWANALYSIS_H

#include "ondrix/Dialect/ondsp/IR/OndspOps.h"

#include "mlir/Dialect/SCF/IR/SCF.h"

#include <cstdint>
#include <optional>

namespace ondrix::analysis {

/// Elements a straight-line form of this reduction would keep live across its
/// enclosing loop's back edge. A sliding operand reads a window that overlaps
/// the next trip's by its length minus the per-trip advance, and unrolling
/// turns exactly that overlap into loop-carried values, which is what decides
/// the shape once the overlap outgrows the target's registers. Zero means
/// nothing slides, so the straight-line form carries nothing.
std::optional<int64_t> getStraightLineCarriedWindow(ondsp::ReduceMacOp reduce);

/// The same quantity for the accumulator loop a reduction has already lowered
/// to, so the two passes that price the straight-line form read one rule
/// rather than two. The window is no longer a view by then: the body reads
/// `base[outer + inner]` from a tensor or from the buffer it bufferizes to.
std::optional<int64_t> getStraightLineCarriedWindow(mlir::scf::ForOp accumulatorLoop);

} // namespace ondrix::analysis

#endif // ONDRIX_ANALYSIS_REDUCTIONWINDOWANALYSIS_H

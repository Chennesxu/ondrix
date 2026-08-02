#ifndef ONDRIX_PIPELINES_ONDRIXPIPELINES_H
#define ONDRIX_PIPELINES_ONDRIXPIPELINES_H

#include "mlir/Pass/PassOptions.h"

#include <cstdint>

namespace mlir {
class OpPassManager;
} // namespace mlir

namespace ondrix {

/// Options of the canonical Ondrix pipeline. The one schedule-relevant knob is
/// a target fact, not a user choice: the vector register width in bits, from
/// which each transform's lane count is derived. Zero disables the schedule
/// stage entirely and yields the ordered scalar program.
struct OndrixDefaultPipelineOptions
    : public mlir::PassPipelineOptions<OndrixDefaultPipelineOptions> {
  Option<int64_t> vectorBits{*this, "vector-bits",
                             llvm::cl::desc("Target vector register width in bits (0 keeps every "
                                            "ordered scalar schedule)"),
                             llvm::cl::init(256)};
};

/// Appends the canonical Ondrix flow to `pm`: contract conversion, boundary
/// bufferization, the automatic schedule stage (candidate transforms filtered
/// by their own legality analyses, applied in the documented priority order),
/// and the lowering tail down to the LLVM dialect. Normal compilation never
/// selects schedules by hand; the individual pass flags remain available for
/// ablation, testing, and oracle runs.
void buildOndrixDefaultPipeline(mlir::OpPassManager &pm,
                                const OndrixDefaultPipelineOptions &options);

/// Registers `-ondrix-default-pipeline` with the global pipeline registry.
void registerOndrixPipelines();

} // namespace ondrix

#endif // ONDRIX_PIPELINES_ONDRIXPIPELINES_H

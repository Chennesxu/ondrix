#ifndef ONDRIX_PIPELINES_ONDRIXPIPELINES_H
#define ONDRIX_PIPELINES_ONDRIXPIPELINES_H

#include "mlir/Pass/PassOptions.h"

#include <cstdint>

namespace mlir {
class OpPassManager;
} // namespace mlir

namespace ondrix {

/// Options of the canonical Ondrix pipeline. Every knob here is a declared
/// target fact rather than a user preference, and none of them is inferred
/// from the machine running the compiler: a cross compiler that reads its own
/// host would silently produce a schedule for the wrong target.
struct OndrixDefaultPipelineOptions
    : public mlir::PassPipelineOptions<OndrixDefaultPipelineOptions> {
  /// Vector register width in bits, from which each transform's lane count is
  /// derived. Zero disables the schedule stage entirely and yields the ordered
  /// scalar program.
  Option<int64_t> vectorBits{*this, "vector-bits",
                             llvm::cl::desc("Target vector register width in bits (0 keeps every "
                                            "ordered scalar schedule)"),
                             llvm::cl::init(256)};
  /// Whether the target has an f32 vector fused multiply-add. This selects
  /// between two schedules that are both inside the declared numeric set, so
  /// a wrong answer costs performance and not conformance. It defaults off
  /// because the penalty is asymmetric: a fused selection on a target without
  /// the instruction becomes one libm call per lane.
  Option<bool> supportsF32VectorFma{
      *this, "supports-f32-vector-fma",
      llvm::cl::desc("Declared target capability: the target has an f32 vector fused "
                     "multiply-add"),
      llvm::cl::init(false)};
};

/// Appends the canonical Ondrix flow to `pm`: design evaluation, contract
/// conversion, forwarding, boundary bufferization, the automatic schedule
/// stage, and the lowering tail down to the LLVM dialect.
void buildOndrixDefaultPipeline(mlir::OpPassManager &pm,
                                const OndrixDefaultPipelineOptions &options);

/// Registers `-ondrix-default-pipeline` with the global pipeline registry.
void registerOndrixPipelines();

} // namespace ondrix

#endif // ONDRIX_PIPELINES_ONDRIXPIPELINES_H

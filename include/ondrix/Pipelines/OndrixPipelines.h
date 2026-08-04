#ifndef ONDRIX_PIPELINES_ONDRIXPIPELINES_H
#define ONDRIX_PIPELINES_ONDRIXPIPELINES_H

#include "mlir/Pass/PassOptions.h"

#include <cstdint>
#include <string>

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
  /// derived. Zero is the default because it is the only value that assumes
  /// nothing: an undeclared target yields the ordered scalar program, which is
  /// legal everywhere. 128 covers NEON and Helium, 256 AVX2.
  Option<int64_t> vectorBits{*this, "vector-bits",
                             llvm::cl::desc("Target vector register width in bits (0, the "
                                            "default, keeps every ordered scalar schedule)"),
                             llvm::cl::init(0)};
  /// Whether the target has an f32 vector fused multiply-add. Both selections
  /// are inside the declared set, so a wrong answer costs performance; off by
  /// default because a fused selection without the instruction becomes one
  /// libm call per lane.
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

/// The canonical flow as the pipeline string `buildOndrixDefaultPipeline`
/// parses. Exposed so a reproduction record can report the schedule that ran
/// rather than a description of it.
std::string getOndrixDefaultPipelineText(const OndrixDefaultPipelineOptions &options);

/// Registers `-ondrix-default-pipeline` with the global pipeline registry.
void registerOndrixPipelines();

} // namespace ondrix

#endif // ONDRIX_PIPELINES_ONDRIXPIPELINES_H

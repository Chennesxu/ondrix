#ifndef ONDRIX_PIPELINES_ONDRIXPIPELINES_H
#define ONDRIX_PIPELINES_ONDRIXPIPELINES_H

#include "mlir/Pass/PassOptions.h"

#include <cstdint>
#include <string>

namespace mlir {
class OpPassManager;
} // namespace mlir

namespace ondrix {

/// Options of the canonical Ondrix pipeline. The target facts here are
/// declared rather than a user preference, and none of them is inferred from
/// the machine running the compiler: a cross compiler that reads its own host
/// would silently produce a schedule for the wrong target. `fftLoops` is the
/// one option that is NOT a target fact, and it says so at its declaration.
struct OndrixDefaultPipelineOptions
    : public mlir::PassPipelineOptions<OndrixDefaultPipelineOptions> {
  /// Vector register width in bits, from which each transform's lane count is
  /// derived. Zero is the default because it is the only value that assumes
  /// nothing: an undeclared target keeps every exact site on the ordered
  /// scalar program, while a fast reduction still carries its multi-chain
  /// rebuild as scalar chains. 128 covers NEON and Helium, 256 AVX2.
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
  /// Which code shape the static transforms take. This is an explicit
  /// SCHEDULE CHOICE, not a target fact, and it must not be derived from a
  /// target description: whether the instruction memory holds the unrolled
  /// transform depends on the extent and on which functions the module
  /// contains, so no global value is right for a module holding both a
  /// size-8 transform (4128 bytes unrolled, and 2.47x faster that way) and a
  /// size-64 one (69352 bytes). It is also not a profitability question the
  /// pipeline could settle: measured on the host class the packed loop form
  /// costs about 2.5x the cycles at extents 8 and 64, while the unrolled form
  /// does not compile at all at extent 1024. Deriving it would need a real
  /// capacity budget and a per-operation decision, which is the cost model
  /// this project refuses.
  Option<bool> fftLoops{*this, "fft-loops",
                        llvm::cl::desc("Schedule choice: lower static CFFT/RFFT/IRFFT as stage "
                                       "loops over in-memory twiddle tables instead of unrolled "
                                       "butterflies (smaller object, slower at small extents)"),
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

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
  /// How many independent partial-sum chains the target's multiply-add
  /// latency needs covered. Zero, the default, derives the count from the
  /// declared width as the measured host classes do; a target whose latency
  /// the width does not predict declares its own, once, for every kernel.
  Option<int64_t> accumulatorChains{
      *this, "accumulator-chains",
      llvm::cl::desc("Declared target quantity: independent partial-sum chains the "
                     "target's multiply-add latency needs (0 derives it from the width)"),
      llvm::cl::init(0)};
  /// The machine's integer register width in bits. A certified constant
  /// reduction groups its products so one group shares a single
  /// accumulator-width add, which is worth an add, a carry test and a second
  /// add on a 32-bit machine and nothing at all on a 64-bit one. 32 is the
  /// default because it assumes the narrower machine and so never withholds
  /// the grouping; a target whose registers already hold the 64-bit carrier
  /// declares it and gets the definitional form on its straight-line
  /// reductions instead.
  Option<int64_t> scalarRegisterBits{
      *this, "scalar-register-bits",
      llvm::cl::desc("Declared target quantity: the machine's integer register width in "
                     "bits (32, the default, assumes the narrower machine)"),
      llvm::cl::init(32)};
  /// How many machine-vector column blocks one matrix output row computes per
  /// iteration. Zero derives it from the width; a target declares its own when
  /// the operand matrix fits its register file and should stay resident.
  Option<int64_t> columnGroup{
      *this, "column-group",
      llvm::cl::desc("Declared target quantity: matrix column blocks per row iteration "
                     "(0 derives it from the width)"),
      llvm::cl::init(0)};
  /// Whether the target's lane-widening integer multiply reads its operands
  /// from the low halves of the wide lanes (x86 pmuldq) rather than from
  /// sign-extended narrower lanes (NEON smull). Both answers are correct
  /// code; the wrong one costs the multiply its instruction selection.
  Option<bool> wideningMultiplyLowHalves{
      *this, "widening-multiply-low-halves",
      llvm::cl::desc("Declared target capability: the lane-widening multiply reads the low "
                     "halves of the wide lanes"),
      llvm::cl::init(true)};
  /// Whether the target's lane-widening integer multiply-add folds adjacent
  /// products (x86 pmaddwd) rather than accumulating each widened product
  /// (NEON smull/saddw). Both answers are correct code; the wrong one costs
  /// a sum of squares two permutes per product vector, or the fold itself.
  Option<bool> multiplyAddAdjacentPairs{
      *this, "multiply-add-adjacent-pairs",
      llvm::cl::desc("Declared target capability: the lane-widening multiply-add folds "
                     "adjacent products"),
      llvm::cl::init(true)};
  /// Whether the target's counted loops run on a zero-overhead hardware
  /// repeat block. The straight-line reduction form buys its speed by
  /// deleting the index update and the branch, and a repeat block has already
  /// deleted both, so unrolling there only spends instruction memory and
  /// denies the block the loop it needs. Off by default: a target without the
  /// instruction pays a branch per term.
  Option<bool> hardwareRepeatBlock{
      *this, "hardware-repeat-block",
      llvm::cl::desc("Declared target capability: counted loops run on a zero-overhead "
                     "hardware repeat block, so reductions keep their loop form"),
      llvm::cl::init(false)};
  /// Values a straight-lined reduction may keep live across its enclosing
  /// loop. A sliding filter carries about one per tap and a dot carries none,
  /// which is the difference the term budget above cannot see: one budget wide
  /// enough for a 64-term dot is far too wide for a 16-tap filter. Zero by
  /// default, because the crossover is a register-file fact no target has
  /// declared yet.
  Option<int64_t> maxCarriedWindow{
      *this, "max-carried-window",
      llvm::cl::desc("Declared target quantity: values a straight-lined reduction may keep "
                     "live across its enclosing loop before the loop form is kept"),
      llvm::cl::init(0)};
  /// Coefficient immediates one constant-row block may materialize before its
  /// terms become a loop over an immutable column-major table. Like `fftLoops`
  /// below this is a SCHEDULE CHOICE and not a target fact: it trades runtime
  /// for instruction memory, and the exchange rate is measured, not derived.
  /// On gem5 AArch64 hpi at 128 bits, the Q15 DCT family (8, 32 and 64 in one
  /// module) runs 1.77x slower under a budget of 128 for an object 1.45x
  /// smaller and codegen 3.1x faster, so zero -- every block straight-line --
  /// is the default and a memory-bounded target opts in.
  Option<int64_t> maxStraightLineCoefficients{
      *this, "max-straight-line-coefficients",
      llvm::cl::desc("Schedule choice: coefficient immediates one constant-row block may "
                     "materialize before its terms become a loop over a column table (0, the "
                     "default, keeps every block straight-line)"),
      llvm::cl::init(0)};
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
  /// Whether the plain-pointer C entries verify their storage preconditions
  /// (no null buffer with elements, no two buffers overlapping) before the
  /// kernel runs. Neither a target fact nor a schedule choice: a debugging
  /// aid for callers, off by default because a valid call pays it for nothing.
  Option<bool> checkedEntries{
      *this, "checked-entries",
      llvm::cl::desc("Make the plain-pointer C entries refuse a null or overlapping buffer "
                     "before touching memory (message and abort)"),
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

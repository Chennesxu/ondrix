#include "ondrix/Pipelines/OndrixPipelines.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

/// The canonical flow as one documented pipeline string. Textual assembly is
/// deliberate: it is byte-comparable with the test RUN lines it generalizes,
/// and the exact string can be recorded in a reproduction manifest.
std::string buildPipelineText(const ondrix::OndrixDefaultPipelineOptions &options) {
  std::string text;
  llvm::raw_string_ostream os(text);

  // Design evaluation must precede conversion (no other pass lowers the
  // design ops); reductions with a direct bufferization stay in contract
  // form so the schedule stage sees their `reduce_mac` loops.
  os << "evaluate-ondrix-fir-design,";
  os << llvm::formatv("convert-ondrix-to-ondsp{{preserve-bufferizable-reductions=true "
                      "output-batch-vector-width={0} fft-loops={1}},",
                      options.vectorBits >= 64 ? options.vectorBits / 32 : 1,
                      options.fftLoops ? "true" : "false");
  // Forwarding must precede bufferization so a forwarded intermediate is
  // never materialized as a buffer.
  os << "canonicalize,cse,forward-ondrix-insert-extract,canonicalize,cse,";
  // A tensor argument is a value: a kernel that updates state in place copies
  // it first and never writes the caller's input.
  os << "declare-ondrix-arguments-read-only,empty-tensor-to-alloc-tensor,";
  // Only a dynamically shaped result still leaves as a fresh buffer under the
  // descriptor ABI; static results become caller buffers below.
  os << "one-shot-bufferize{bufferize-function-boundaries=true allow-return-allocs=true "
        "function-boundary-type-conversion=identity-layout-map create-deallocs=false},";
  os << "cse,canonicalize,";
  // The C entry is recorded here, while the reductions that pair equal-length
  // windows are still `reduce_mac`; it is built once the descriptors expand.
  os << "declare-ondrix-c-entry-points,";

  // The schedule stage; each transform is its own legality filter. Order is
  // the policy: vertical batchings first, then horizontal reductions,
  // saturating before wrap so certified prefixes reach the route with proofs.
  if (options.vectorBits >= 64) {
    int64_t lanes = options.vectorBits / 32;
    // Measured per-target defaults: the 256-bit host class pays for eight
    // chains, the 128-bit in-order class regresses under them; two column
    // blocks per row keep an 8x8 operand register-resident on both.
    int64_t columnGroup = options.columnGroup > 0 ? options.columnGroup : 2;
    int64_t chainDepth = options.accumulatorChains > 0 ? options.accumulatorChains
                                                       : (options.vectorBits >= 256 ? 8 : 4);
    os << llvm::formatv("vectorize-ondsp-fp-filter-outputs{{vector-width={0} "
                        "supports-vector-fma={1} interleave=4 column-group={2} "
                        "row-horizontal={3}},",
                        lanes, options.supportsF32VectorFma ? "true" : "false", columnGroup,
                        options.vectorBits >= 256 ? "true" : "false");
    // Chain count: derived from the width as a host-class heuristic for the
    // FMA latency-throughput product unless the target declares its own
    // (--accumulator-chains). The pass clamps to the block count per site.
    os << llvm::formatv("vectorize-ondsp-fp-fast-memref-reduce{{vector-width={0} "
                        "supports-vector-fma={1} interleave={2}},",
                        lanes, options.supportsF32VectorFma ? "true" : "false", chainDepth);
    // Two output vectors per sliding-window block: four spill their per-lane
    // i64 accumulators on a 128-bit file. Requantized-product sites batch only
    // from 256 bits; the option's description carries the measurement.
    os << llvm::formatv("vectorize-ondsp-fixed-decimate-outputs{{vector-width={0} chunk-multiple=2 "
                        "requantized-products={1}},",
                        lanes, options.vectorBits >= 256 ? "true" : "false");
    os << llvm::formatv("vectorize-ondsp-fixed-elementwise-updates{{vector-width={0}},", lanes);
    os << llvm::formatv("vectorize-ondsp-fixed-elementwise-loops{{vector-width={0}},", lanes);
    // The convolution-shaped reduction its operands walk in opposite
    // directions cannot reach the reduce_mac routes below, which pair their
    // operands in increasing index order on both sides.
    os << llvm::formatv("vectorize-ondsp-fixed-window-mac-reduce{{vector-width={0}},", lanes);
    // Four machine vectors per certified chunk: like the interleave above this
    // is a host-class heuristic, not a target fact. Both ladders fall back per
    // reduction, so the only thing a wrong guess costs is a narrower chunk.
    os << llvm::formatv(
        "vectorize-ondsp-constant-saturating-memref-reduce{{vector-width={0} chunk-multiple=4 "
        "max-elements=64},",
        lanes);
    os << llvm::formatv("vectorize-ondsp-fixed-memref-reduce{{vector-width={0} chunk-multiple=4 "
                        "pair-fold-squares={1} requantized-products={2}},",
                        lanes, options.multiplyAddAdjacentPairs ? "true" : "false",
                        options.vectorBits >= 256 ? "true" : "false");
    os << "parallelize-ondsp-fixed-wrap-vector-reduce,";
    os << "normalize-ondsp-fixed-vector-reduce,";
  } else {
    // Chain count is ILP, not lane count: the multi-chain rebuild pays on a
    // core with no usable lanes, so it is not gated on the SIMD stage.
    os << llvm::formatv("vectorize-ondsp-fp-fast-memref-reduce{{vector-width=1 "
                        "supports-vector-fma={0} interleave={1}},",
                        options.supportsF32VectorFma ? "true" : "false",
                        options.accumulatorChains > 0 ? options.accumulatorChains : 4);
  }

  // Unrolling is ILP, not lanes: it runs at every width, after the batchers,
  // on their residue. The 128-term budget is a host-class measurement (pass
  // description); under a declared repeat block counted reductions keep loops.
  os << "forward-ondsp-packed-reduction-operands,";
  int64_t straightLineTerms = options.hardwareRepeatBlock ? 1 : 128;
  os << llvm::formatv("scalarize-ondsp-fixed-reduce-mac{{max-unrolled-terms={0}},"
                      "unroll-ondsp-fixed-mac-loops{{max-unrolled-terms={0}},",
                      straightLineTerms);
  // The f32 sibling under the same budget; above one lane the lane-blocked
  // ordered lowering is the better claim on a reduction, so only the
  // accumulator loops are left for it.
  os << llvm::formatv("unroll-ondsp-fp-ordered-reduce{{vector-width={0} "
                      "max-straight-line-terms={1} max-unrolled-terms={2}},",
                      options.vectorBits >= 64 ? options.vectorBits / 32 : 1,
                      options.hardwareRepeatBlock ? 1 : 256, options.hardwareRepeatBlock ? 1 : 512);

  // Lowering tail down to the LLVM dialect. The declared-off reduction batches
  // its products at the target width; the fold order is untouched, so this is
  // reached whether or not the schedule stage ran.
  if (options.vectorBits >= 64)
    os << llvm::formatv("lower-ondsp-f32-reduce-to-scalar{{vector-width={0}},",
                        options.vectorBits / 32);
  else
    os << "lower-ondsp-f32-reduce-to-scalar,";
  // Static results are written into caller buffers, no copy: the C convention
  // declares an output disjoint from every other argument (distinct-out-params).
  os << "convert-ondrix-static-results-to-out-params,"
        "forward-ondrix-result-buffers{distinct-out-params=true},"
        "lower-memref-copy-to-scf,";
  os << llvm::formatv("convert-ondsp-fixed-to-scalar{{widening-multiply-low-halves={0}},",
                      options.wideningMultiplyLowHalves ? "true" : "false");
  // Small kernel-local temporaries live on the stack; the C baselines never pay
  // a heap round trip per call, so the compiled kernels must not either.
  os << "func.func(promote-buffers-to-stack,buffer-deallocation),";
  os << "convert-vector-to-scf,expand-strided-metadata,lower-affine,convert-scf-to-cf,"
        "convert-vector-to-llvm,"
        "finalize-memref-to-llvm,convert-math-to-llvm,convert-arith-to-llvm,convert-cf-to-llvm,"
        "convert-func-to-llvm,apply-ondrix-llvm-argument-attributes,reconcile-unrealized-casts,"
        "emit-ondrix-c-entry-points";
  if (options.checkedEntries)
    os << "{checked=true}";
  return text;
}

} // namespace

std::string ondrix::getOndrixDefaultPipelineText(const OndrixDefaultPipelineOptions &options) {
  return buildPipelineText(options);
}

void ondrix::buildOndrixDefaultPipeline(OpPassManager &pm,
                                        const OndrixDefaultPipelineOptions &options) {
  std::string text = buildPipelineText(options);
  if (failed(parsePassPipeline(text, pm)))
    llvm::report_fatal_error(llvm::Twine("invalid ondrix default pipeline: ") + text);
}

void ondrix::registerOndrixPipelines() {
  PassPipelineRegistration<OndrixDefaultPipelineOptions>(
      "ondrix-default-pipeline",
      "The canonical Ondrix flow: contract conversion, boundary bufferization, "
      "the automatic legality-filtered schedule stage, and lowering to the "
      "LLVM dialect",
      buildOndrixDefaultPipeline);
}

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
  os << "convert-ondrix-to-ondsp{preserve-bufferizable-reductions=true},";
  // Forwarding must precede bufferization so a forwarded intermediate is
  // never materialized as a buffer.
  os << "canonicalize,cse,forward-ondrix-insert-extract,canonicalize,cse,";
  os << "empty-tensor-to-alloc-tensor,";
  // Entry points may return fresh result buffers; ownership transfers to the
  // caller, which is the documented (unstable) descriptor ABI the harnesses
  // already follow. The deallocation pass below frees only what does not
  // escape.
  os << "one-shot-bufferize{bufferize-function-boundaries=true allow-return-allocs=true "
        "function-boundary-type-conversion=identity-layout-map},";
  os << "cse,canonicalize,";

  // The automatic schedule stage. Every candidate transform is fail-closed —
  // its own legality analysis is the filter, and an unauthorized site keeps
  // the ordered schedule. The fixed order is the selection policy: the
  // order-preserving vertical batchings run first because a batched block
  // serves W outputs per pass over the taps, then the horizontal reductions
  // take the sites vertical batching cannot (single reductions and ordered
  // remainders), saturating before wrap so certified prefixes are consumed
  // by the route that carries their proofs. Lane counts are derived from the
  // target width at the Q15-product/f32 element size; a Q31 site simply uses
  // wider registers.
  if (options.vectorBits >= 64) {
    int64_t lanes = options.vectorBits / 32;
    // Column grouping and chain depth are measured per-target policies: the
    // 256-bit host class pays for grouping and for eight chains, while the
    // 128-bit in-order class regresses under both (its load pipe serializes
    // the wider schedules' paired accesses).
    int64_t columnGroup = options.vectorBits >= 256 ? 2 : 1;
    int64_t chainDepth = options.vectorBits >= 256 ? 8 : 4;
    os << llvm::formatv("vectorize-ondsp-fp-filter-outputs{{vector-width={0} "
                        "supports-vector-fma={1} interleave=4 column-group={2} "
                        "row-horizontal={3}},",
                        lanes, options.supportsF32VectorFma ? "true" : "false", columnGroup,
                        options.vectorBits >= 256 ? "true" : "false");
    // Four chains: a host-class heuristic for the FMA latency-throughput
    // product, not a target fact; the pass clamps to the block count per
    // site, and a target schedule parameter can override it later.
    os << llvm::formatv("vectorize-ondsp-fp-fast-memref-reduce{{vector-width={0} "
                        "supports-vector-fma={1} interleave={2}},",
                        lanes, options.supportsF32VectorFma ? "true" : "false", chainDepth);
    os << llvm::formatv("vectorize-ondsp-fixed-decimate-outputs{{vector-width={0}},", lanes);
    os << llvm::formatv("vectorize-ondsp-fixed-elementwise-updates{{vector-width={0}},", lanes);
    // Four machine vectors per certified chunk: like the interleave above this
    // is a host-class heuristic, not a target fact. Both ladders fall back per
    // reduction, so the only thing a wrong guess costs is a narrower chunk.
    os << llvm::formatv(
        "vectorize-ondsp-constant-saturating-memref-reduce{{vector-width={0} chunk-multiple=4 "
        "max-elements=64},",
        lanes);
    os << llvm::formatv("vectorize-ondsp-fixed-memref-reduce{{vector-width={0} chunk-multiple=4},",
                        lanes);
    os << "parallelize-ondsp-fixed-wrap-vector-reduce,";
    os << "normalize-ondsp-fixed-vector-reduce,";
  } else {
    // Chain count is ILP, not lane count: the multi-chain rebuild pays on a
    // core with no usable lanes, so it is not gated on the SIMD stage.
    os << llvm::formatv("vectorize-ondsp-fp-fast-memref-reduce{{vector-width=1 "
                        "supports-vector-fma={0} interleave=4},",
                        options.supportsF32VectorFma ? "true" : "false");
  }

  // Straight-line short chains are ILP, not lane count: dropping the per-term
  // index update and branch pays on a core with no usable lanes, so like the
  // chain rebuild above this is not gated on the SIMD stage. Both run after
  // the batchers, so a rewritten body can never hide a shape they match; what
  // reaches them is the residue no batcher claimed.
  // The budget is a host-class measurement, not a target fact: at 32 terms a
  // filter tap chain runs 1.9x faster for 2.7x its bytes, while a DCT's
  // per-row replication reaches 4096 terms and 15x its bytes to run SLOWER.
  os << "scalarize-ondsp-fixed-reduce-mac{max-unrolled-terms=128},"
        "unroll-ondsp-fixed-mac-loops{max-unrolled-terms=128},";

  // Lowering tail down to the LLVM dialect. The declared-off reduction batches
  // its products at the target width; the fold order is untouched, so this is
  // reached whether or not the schedule stage ran.
  if (options.vectorBits >= 64)
    os << llvm::formatv("lower-ondsp-f32-reduce-to-scalar{{vector-width={0}},",
                        options.vectorBits / 32);
  else
    os << "lower-ondsp-f32-reduce-to-scalar,";
  os << "lower-rank-one-memref-copy-to-scf,";
  os << "convert-ondsp-fixed-to-scalar,";
  os << "func.func(buffer-deallocation),";
  os << "expand-strided-metadata,lower-affine,convert-scf-to-cf,convert-vector-to-llvm,"
        "finalize-memref-to-llvm,convert-math-to-llvm,convert-arith-to-llvm,convert-cf-to-llvm,"
        "convert-func-to-llvm,reconcile-unrealized-casts";
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

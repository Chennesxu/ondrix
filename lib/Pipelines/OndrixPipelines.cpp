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
    os << llvm::formatv("vectorize-ondsp-fp-filter-outputs{{vector-width={0} "
                        "supports-vector-fma={1}},",
                        lanes, options.supportsF32VectorFma ? "true" : "false");
    // Four chains: a host-class heuristic for the FMA latency-throughput
    // product, not a target fact; the pass clamps to the block count per
    // site, and a target schedule parameter can override it later.
    os << llvm::formatv("vectorize-ondsp-fp-fast-memref-reduce{{vector-width={0} "
                        "supports-vector-fma={1} interleave=4},",
                        lanes, options.supportsF32VectorFma ? "true" : "false");
    os << llvm::formatv("vectorize-ondsp-fixed-decimate-outputs{{vector-width={0}},", lanes);
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
  }

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

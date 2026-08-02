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
std::string defaultPipelineText(const ondrix::OndrixDefaultPipelineOptions &options) {
  std::string text;
  llvm::raw_string_ostream os(text);

  // Contract conversion. Compile-time design intents are evaluated first —
  // the fail-closed quantization tie guard aborts the compile rather than
  // ship a misquantized table. Algorithm operations whose reductions have a
  // direct bufferization stay in contract form here so the schedule stage
  // below sees their `reduce_mac` loops; everything else lowers to its ondsp
  // event form.
  os << "evaluate-ondrix-fir-design,";
  os << "convert-ondrix-to-ondsp{preserve-bufferizable-reductions=true},";
  // Proof-gated forwarding joins the automatic flow: constant-index reads
  // staged through insert chains (an FFT feeding magnitude) collapse onto
  // the producing SSA values, deleting the packed intermediate before
  // bufferization ever materializes it. Fail-closed and inert elsewhere.
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
    os << llvm::formatv("vectorize-ondsp-fp-filter-outputs{{vector-width={0}},", lanes);
    os << llvm::formatv("vectorize-ondsp-fixed-decimate-outputs{{vector-width={0}},", lanes);
    os << llvm::formatv(
        "vectorize-ondsp-constant-saturating-memref-reduce{{vector-width={0} max-elements=64},",
        lanes);
    os << llvm::formatv("vectorize-ondsp-fixed-memref-reduce{{vector-width={0}},", lanes);
    os << "parallelize-ondsp-fixed-wrap-vector-reduce,";
    os << "normalize-ondsp-fixed-vector-reduce,";
  }

  // Lowering tail down to the LLVM dialect.
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

void ondrix::buildOndrixDefaultPipeline(OpPassManager &pm,
                                        const OndrixDefaultPipelineOptions &options) {
  std::string text = defaultPipelineText(options);
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

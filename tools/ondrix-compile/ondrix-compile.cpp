#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"
#include "ondrix/Frontend/OxFrontend.h"
#include "ondrix/InitAllDialects.h"
#include "ondrix/InitAllPasses.h"
#include "ondrix/Pipelines/OndrixPipelines.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/Config/llvm-config.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;

namespace {
enum class EmitKind { Contracts, LLVMDialect, Manifest };

cl::opt<std::string> inputFilename(cl::Positional, cl::desc("<input .ox file>"), cl::Required);
cl::opt<std::string> outputFilename("o", cl::desc("Output MLIR file"), cl::value_desc("filename"),
                                    cl::init("-"));
cl::opt<bool> printSourceLocations("print-source-locations",
                                   cl::desc("Print source locations in emitted MLIR"));
cl::opt<EmitKind> emitKind(
    "emit", cl::desc("What to emit"),
    cl::values(clEnumValN(EmitKind::Contracts, "contracts",
                          "The frontend's contract-form MLIR (default)"),
               clEnumValN(EmitKind::LLVMDialect, "llvm",
                          "LLVM-dialect MLIR produced by the canonical pipeline, "
                          "schedules selected automatically under their legality analyses"),
               clEnumValN(EmitKind::Manifest, "manifest",
                          "JSON reproduction record of the compilation that "
                          "--emit=llvm would perform")),
    cl::init(EmitKind::Contracts));

// Target facts for the schedule stage. Both default to assuming nothing, so an
// undeclared target compiles to the ordered program rather than to a guess.
cl::opt<int64_t> vectorBits("vector-bits",
                            cl::desc("Target vector register width in bits (0 keeps exact "
                                     "sites ordered and scalar; fast reductions still carry "
                                     "scalar chains)"),
                            cl::init(0));
cl::opt<bool> supportsF32VectorFma("supports-f32-vector-fma",
                                   cl::desc("Declared target capability: the target has an f32 "
                                            "vector fused multiply-add"),
                                   cl::init(false));
} // namespace

/// The decisions this compilation made, in the compiler's own terms. Not a
/// reproduction record: the environment around it — git revision, the llc
/// invocation, reference compiler flags, corpus seeds, object hashes — is the
/// harness's to add, and inventing empty fields for them here would read as
/// though something had checked them.
///
/// The permission set is the compilation's, not any one site's. A source
/// operation reaching two mechanisms is summarized rather than broken out;
/// per-site attribution is deferred with the rest of the contract verification
/// work (docs/f32-contract-evidence.md).
void emitManifest(mlir::ModuleOp module, const ondrix::OndrixDefaultPipelineOptions &options,
                  raw_ostream &os) {
  llvm::json::Array permissions;
  if (auto spent =
          module->getAttrOfType<mlir::ArrayAttr>(ondrix::ondsp::getFastPermissionAttrName()))
    for (mlir::Attribute entry : spent)
      permissions.push_back(mlir::cast<mlir::StringAttr>(entry).getValue());

  const int64_t vectorBitsValue = options.vectorBits;
  const bool fmaValue = options.supportsF32VectorFma;
  llvm::json::Object manifest{
      {"llvm_version", LLVM_VERSION_STRING},
      {"pipeline", ondrix::getOndrixDefaultPipelineText(options)},
      {"declared_target_facts",
       llvm::json::Object{{"vector_bits", vectorBitsValue}, {"supports_f32_vector_fma", fmaValue}}},
      {"fast_permissions_used", std::move(permissions)},
      // Declared, not observed: the numeric model states these and every
      // reference is built to match them.
      {"required_fp_environment", llvm::json::Object{{"rounding", "round_to_nearest_even"},
                                                     {"subnormals", "preserved"},
                                                     {"flush_to_zero", false},
                                                     {"exception_state", "unobservable"}}},
  };
  os << llvm::formatv("{0:2}", llvm::json::Value(std::move(manifest))) << '\n';
}

int main(int argc, char **argv) {
  InitLLVM initLLVM(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "Ondrix .ox frontend\n");

  ErrorOr<std::unique_ptr<MemoryBuffer>> input = MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (!input) {
    errs() << "ondrix-compile: cannot read '" << inputFilename
           << "': " << input.getError().message() << "\n";
    return 1;
  }

  std::error_code outputError;
  ToolOutputFile output(outputFilename, outputError, sys::fs::OF_Text);
  if (outputError) {
    errs() << "ondrix-compile: cannot open '" << outputFilename << "': " << outputError.message()
           << "\n";
    return 1;
  }

  // The canonical pipeline is assembled from registered pass names, so both
  // the pass registry and the full dialect set must be present even though
  // the frontend itself needs neither.
  mlir::registerAllPasses();
  ondrix::registerAllOndrixPasses();
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  ondrix::registerAllOndrixDialects(registry);

  mlir::MLIRContext context;
  context.appendDialectRegistry(registry);
  auto module =
      ondrix::frontend::compileOxSource(inputFilename, (*input)->getBuffer(), context, errs());
  if (!module)
    return 1;

  if (emitKind == EmitKind::LLVMDialect || emitKind == EmitKind::Manifest) {
    mlir::PassManager passManager(&context, mlir::ModuleOp::getOperationName());
    ondrix::OndrixDefaultPipelineOptions options;
    options.vectorBits = vectorBits.getValue();
    options.supportsF32VectorFma = supportsF32VectorFma.getValue();
    ondrix::buildOndrixDefaultPipeline(passManager, options);
    if (failed(passManager.run(*module)))
      return 1;
    if (emitKind == EmitKind::Manifest) {
      emitManifest(*module, options, output.os());
      output.keep();
      return 0;
    }
  }

  mlir::OpPrintingFlags flags;
  if (printSourceLocations)
    flags.enableDebugInfo(/*enable=*/true, /*prettyForm=*/false);
  module->print(output.os(), flags);
  output.os() << '\n';
  output.keep();
  return 0;
}

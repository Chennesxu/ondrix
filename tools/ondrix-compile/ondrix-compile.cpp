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
                            cl::desc("Target vector register width in bits (0 keeps every "
                                     "ordered scalar schedule)"),
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
void emitManifest(mlir::ModuleOp module, const ondrix::OndrixDefaultPipelineOptions &options,
                  raw_ostream &os) {
  llvm::json::Array permissions;
  if (auto spent =
          module->getAttrOfType<mlir::ArrayAttr>(ondrix::ondsp::getFastPermissionAttrName()))
    for (mlir::Attribute entry : spent)
      permissions.push_back(mlir::cast<mlir::StringAttr>(entry).getValue());

  // Per static selection site, because the set above is true of the
  // compilation and false of every site in it: one full-boundary filter
  // generates guarded ordered edges and a horizontal interior, and a site with
  // a dynamic extent generates one case per branch.
  llvm::json::Array sites;
  if (auto records =
          module->getAttrOfType<mlir::ArrayAttr>(ondrix::ondsp::getFastSelectionAttrName()))
    for (mlir::Attribute entry : records) {
      auto record = mlir::cast<mlir::DictionaryAttr>(entry);
      auto text = [&](llvm::StringRef field) {
        return record.getAs<mlir::StringAttr>(field).getValue();
      };
      llvm::json::Array used;
      for (mlir::Attribute name : record.getAs<mlir::ArrayAttr>("used_permissions"))
        used.push_back(mlir::cast<mlir::StringAttr>(name).getValue());
      llvm::json::Object site{
          {"source_site_id", text("source_site_id")},
          {"source_operation", text("source_operation")},
          {"route_role", text("route_role")},
          {"instance_domain", text("instance_domain")},
          {"mechanism", text("mechanism")},
          {"used_permissions", std::move(used)},
      };
      // Omitted rather than empty for an unconditional site: an empty
      // condition would read as a condition that was checked.
      if (!text("when").empty())
        site["when"] = text("when");
      sites.push_back(std::move(site));
    }

  const int64_t vectorBitsValue = options.vectorBits;
  const bool fmaValue = options.supportsF32VectorFma;
  llvm::json::Object manifest{
      {"llvm_version", LLVM_VERSION_STRING},
      {"pipeline", ondrix::getOndrixDefaultPipelineText(options)},
      {"declared_target_facts",
       llvm::json::Object{{"vector_bits", vectorBitsValue}, {"supports_f32_vector_fma", fmaValue}}},
      {"fast_permissions_used", std::move(permissions)},
      {"fast_selection_sites", std::move(sites)},
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

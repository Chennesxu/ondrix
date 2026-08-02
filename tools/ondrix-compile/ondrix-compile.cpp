#include "ondrix/Frontend/OxFrontend.h"
#include "ondrix/InitAllDialects.h"
#include "ondrix/InitAllPasses.h"
#include "ondrix/Pipelines/OndrixPipelines.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;

namespace {
enum class EmitKind { Contracts, LLVMDialect };

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
                          "schedules selected automatically under their legality analyses")),
    cl::init(EmitKind::Contracts));
} // namespace

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

  if (emitKind == EmitKind::LLVMDialect) {
    mlir::PassManager passManager(&context, mlir::ModuleOp::getOperationName());
    ondrix::OndrixDefaultPipelineOptions options;
    ondrix::buildOndrixDefaultPipeline(passManager, options);
    if (failed(passManager.run(*module)))
      return 1;
  }

  mlir::OpPrintingFlags flags;
  if (printSourceLocations)
    flags.enableDebugInfo(/*enable=*/true, /*prettyForm=*/false);
  module->print(output.os(), flags);
  output.os() << '\n';
  output.keep();
  return 0;
}

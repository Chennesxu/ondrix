#include "ondrix/Frontend/OxFrontend.h"

#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;

namespace {
cl::opt<std::string> inputFilename(cl::Positional, cl::desc("<input .ox file>"), cl::Required);
cl::opt<std::string> outputFilename("o", cl::desc("Output MLIR file"), cl::value_desc("filename"),
                                    cl::init("-"));
cl::opt<bool> printSourceLocations("print-source-locations",
                                   cl::desc("Print source locations in emitted MLIR"));
} // namespace

int main(int argc, char **argv) {
  InitLLVM initLLVM(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "Ondrix .ox frontend\n");

  ErrorOr<std::unique_ptr<MemoryBuffer>> input = MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (!input) {
    errs() << "ondrixc: cannot read '" << inputFilename << "': " << input.getError().message()
           << "\n";
    return 1;
  }

  std::error_code outputError;
  ToolOutputFile output(outputFilename, outputError, sys::fs::OF_Text);
  if (outputError) {
    errs() << "ondrixc: cannot open '" << outputFilename << "': " << outputError.message() << "\n";
    return 1;
  }

  mlir::MLIRContext context;
  auto module =
      ondrix::frontend::compileOxSource(inputFilename, (*input)->getBuffer(), context, errs());
  if (!module)
    return 1;

  mlir::OpPrintingFlags flags;
  if (printSourceLocations)
    flags.enableDebugInfo(/*enable=*/true, /*prettyForm=*/false);
  module->print(output.os(), flags);
  output.os() << '\n';
  output.keep();
  return 0;
}

#include "ondrix/InitAllDialects.h"
#include "ondrix/InitAllPasses.h"
#include "ondrix/Pipelines/OndrixPipelines.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  ondrix::registerAllOndrixPasses();
  ondrix::registerOndrixPipelines();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  ondrix::registerAllOndrixDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "ondrix optimizer driver\n", registry));
}

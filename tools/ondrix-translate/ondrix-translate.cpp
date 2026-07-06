#include "ondrix/InitAllTranslations.h"

#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

int main(int argc, char **argv) {
  mlir::registerAllTranslations();
  ondrix::registerAllOndrixTranslations();

  return failed(
      mlir::mlirTranslateMain(argc, argv, "ondrix translation driver\n"));
}


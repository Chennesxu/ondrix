#include "ondrix/InitAllDialects.h"

#include "ondrix/Dialect/ondrix/IR/OndrixDialect.h"
#include "ondrix/Dialect/ondrix/Transforms/BufferizableOpInterfaceImpl.h"
#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreDialect.h"

#include "mlir/IR/DialectRegistry.h"

using namespace mlir;

void ondrix::registerAllOndrixDialects(DialectRegistry &registry) {
  registry.insert<ondrix::ir::OndrixDialect, ondrix::ondsp::OndspDialect,
                  ondrix::ortumcore::OrtumCoreDialect>();
  ondrix::ir::registerBufferizableOpInterfaceExternalModels(registry);
}

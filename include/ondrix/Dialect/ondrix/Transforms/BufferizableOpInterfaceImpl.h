#ifndef ONDRIX_DIALECT_ONDRIX_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H
#define ONDRIX_DIALECT_ONDRIX_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H

namespace mlir {
class DialectRegistry;
}

namespace ondrix::ir {

void registerBufferizableOpInterfaceExternalModels(mlir::DialectRegistry &registry);

} // namespace ondrix::ir

#endif // ONDRIX_DIALECT_ONDRIX_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H

#ifndef ONDRIX_CONVERSION_ONDSPTOORTUMCORE_ONDSPTOORTUMCORE_H
#define ONDRIX_CONVERSION_ONDSPTOORTUMCORE_ONDSPTOORTUMCORE_H

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace ondrix {

std::unique_ptr<mlir::Pass> createConvertOndspToOrtumCorePass();
void registerConvertOndspToOrtumCorePass();

} // namespace ondrix

#endif

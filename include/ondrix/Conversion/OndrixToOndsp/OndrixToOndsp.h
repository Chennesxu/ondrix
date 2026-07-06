#ifndef ONDRIX_CONVERSION_ONDRIXTOONDSP_ONDRIXTOONDSP_H
#define ONDRIX_CONVERSION_ONDRIXTOONDSP_ONDRIXTOONDSP_H

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace ondrix {

std::unique_ptr<mlir::Pass> createConvertOndrixToOndspPass();
void registerConvertOndrixToOndspPass();
void registerConversionPasses();

} // namespace ondrix

#endif

#ifndef ONDRIX_INITALLDIALECTS_H
#define ONDRIX_INITALLDIALECTS_H

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace ondrix {

void registerAllOndrixDialects(mlir::DialectRegistry &registry);

} // namespace ondrix

#endif // ONDRIX_INITALLDIALECTS_H


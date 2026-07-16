#ifndef ONDRIX_CONVERSION_UTILS_STRUCTURALTYPECONVERSIONS_H
#define ONDRIX_CONVERSION_UTILS_STRUCTURALTYPECONVERSIONS_H

namespace mlir {
class RewritePatternSet;
class TypeConverter;
} // namespace mlir

namespace ondrix::conversion {

/// Populates type-only rewrites for host operations that commonly carry
/// converted accumulator values without owning their numeric semantics.
void populateCommonStructuralTypeConversionPatterns(mlir::TypeConverter &typeConverter,
                                                    mlir::RewritePatternSet &patterns);

} // namespace ondrix::conversion

#endif // ONDRIX_CONVERSION_UTILS_STRUCTURALTYPECONVERSIONS_H

#ifndef ONDRIX_CONVERSION_UTILS_VALUETYPECONVERSIONS_H
#define ONDRIX_CONVERSION_UTILS_VALUETYPECONVERSIONS_H

namespace mlir {
class RewritePatternSet;
class TypeConverter;
} // namespace mlir

namespace ondrix::conversion {

/// Populates type-only rewrites for host operations that transport converted
/// SSA values without owning their semantics.
void populateValueTypeConversionPatterns(mlir::TypeConverter &typeConverter,
                                         mlir::RewritePatternSet &patterns);

} // namespace ondrix::conversion

#endif // ONDRIX_CONVERSION_UTILS_VALUETYPECONVERSIONS_H

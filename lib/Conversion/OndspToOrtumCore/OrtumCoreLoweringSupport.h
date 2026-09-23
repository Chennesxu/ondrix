#ifndef ONDRIX_LIB_CONVERSION_ONDSPTOORTUMCORE_ORTUMCORELOWERINGSUPPORT_H
#define ONDRIX_LIB_CONVERSION_ONDSPTOORTUMCORE_ORTUMCORELOWERINGSUPPORT_H

#include "ondrix/Dialect/ondsp/IR/OndspAttrs.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspTypes.h"
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreEnums.h"
#include "ondrix/Target/OrtumCore/OrtumCoreTargetProfile.h"

#include "mlir/IR/Builders.h"

#include <optional>

namespace ondrix::conversion {

/// Maps one lane of an ondsp accumulator onto the target accumulator domain.
/// The lane count is deliberately absent: single-lane and lane-pair producers
/// share the same per-lane admission and differ only in how many target
/// accumulator values they thread.
ortumcore::AccumulatorDomain getOrtumCoreAccumulatorDomain(ondsp::AccType accumulator);

/// True when one lane of `accumulator` is the proven target accumulator.
bool isOrtumCoreLaneDomain(ondsp::AccType accumulator);

/// True when the mac policy is the proven signed Q15 full-product capability
/// on the target lane domain. Silent on every rejection.
bool isOrtumCoreMacPolicy(ondsp::MacOp op);

/// One admitted readout realization of an `acc_export` policy.
struct OrtumCoreExportPolicy {
  int64_t shift;
  mlir::IntegerType storage;
  ondsp::RoundingMode rounding;
};

/// Classifies the export policy against the proven readout capability, or
/// nullopt when it falls outside. Result element/lane shape stays with the
/// caller. Silent on every rejection.
std::optional<OrtumCoreExportPolicy> classifyOrtumCoreExport(ondsp::AccExportOp op);

/// The one capability readout shift the composition for `policy` reads.
int64_t getOrtumCoreReadoutShift(const OrtumCoreExportPolicy &policy);

/// Emits the proven readout composition for one converted accumulator value
/// and returns the exported lane in the destination storage type. The
/// exactness argument lives on the ConvertOndspToOrtumCore description.
mlir::Value emitOrtumCoreReadout(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value acc,
                                 const OrtumCoreExportPolicy &policy);

/// The same composition over any source whose `readout(shift)` equals the
/// capability readout `sat32(acc >> shift)` at the shift the policy reads.
mlir::Value emitOrtumCoreReadout(mlir::OpBuilder &builder, mlir::Location loc,
                                 llvm::function_ref<mlir::Value(int64_t)> readout,
                                 const OrtumCoreExportPolicy &policy);

/// The packed target rounding inventory. nearest_even and toward_zero
/// deliberately map to nothing so those profiles stay on the generic path,
/// and a newly declared mode lands there too (plus a -Wswitch finding in the
/// definition) instead of borrowing an inventory member.
std::optional<ortumcore::CxRounding> selectPackedComplexRounding(ondsp::RoundingMode mode);

/// Clamps a signed i32 into `storage` and truncates; the identity at i32.
/// Clamping to a wider range first cannot change a narrower clamp, which is
/// what lets a readout compose with this.
mlir::Value emitSignedSaturatingNarrow(mlir::OpBuilder &builder, mlir::Location loc,
                                       mlir::Value value, mlir::IntegerType storage);

} // namespace ondrix::conversion

#endif // ONDRIX_LIB_CONVERSION_ONDSPTOORTUMCORE_ORTUMCORELOWERINGSUPPORT_H

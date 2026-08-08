#ifndef ONDRIX_LIB_CONVERSION_ONDSPTOORTUMCORE_ORTUMCORELOWERINGSUPPORT_H
#define ONDRIX_LIB_CONVERSION_ONDSPTOORTUMCORE_ORTUMCORELOWERINGSUPPORT_H

#include "ondrix/Dialect/ondsp/IR/OndspAttrs.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspTypes.h"
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

/// Emits the proven readout composition for one converted accumulator value
/// and returns the exported lane in the destination storage type. The
/// exactness argument lives on the ConvertOndspToOrtumCore description.
mlir::Value emitOrtumCoreReadout(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value acc,
                                 const OrtumCoreExportPolicy &policy);

} // namespace ondrix::conversion

#endif // ONDRIX_LIB_CONVERSION_ONDSPTOORTUMCORE_ORTUMCORELOWERINGSUPPORT_H

#ifndef ONDRIX_SUPPORT_DCTCOEFFICIENTS_H
#define ONDRIX_SUPPORT_DCTCOEFFICIENTS_H

#include "ondrix/Support/GuardedQ15Quantization.h"

#include <cmath>
#include <cstdint>
#include <optional>

namespace ondrix {

// Type-II DCT coefficient c[k][n] = q15(cos(pi*(2n+1)*k/(2N))) under the same
// tie-guarded round-half-even quantization as the twiddle tables. This is the
// single table generator shared by every `ondrix.dct` consumer — the tensor
// lowering and the bufferization interface must never derive the table twice.
// A 50-digit sweep of all 5456 coefficients for the supported extents shows a
// worst tie margin of 0.0044 LSB (the 124 saturations are the k = 0 rows, by
// declared convention), so all supported extents are admissible; the guard
// remains the fail-closed backstop.
inline std::optional<int64_t> getDctCoefficientQ15(int64_t extent, int64_t k, int64_t n) {
  constexpr double kPi = 3.14159265358979323846264338327950288;
  double angle = kPi * static_cast<double>((2 * n + 1) * k) / (2.0 * static_cast<double>(extent));
  std::optional<GuardedQ15Value> quantized = quantizeGuardedQ15(std::cos(angle));
  if (!quantized)
    return std::nullopt;
  return quantized->value;
}

// Fail-closed admissibility of the complete coefficient matrix of one static
// extent. A consumer checks this once and may then rely on every individual
// coefficient query succeeding.
inline bool hasAdmissibleDctCoefficients(int64_t extent) {
  for (int64_t k = 0; k < extent; ++k)
    for (int64_t n = 0; n < extent; ++n)
      if (!getDctCoefficientQ15(extent, k, n))
        return false;
  return true;
}

} // namespace ondrix

#endif // ONDRIX_SUPPORT_DCTCOEFFICIENTS_H

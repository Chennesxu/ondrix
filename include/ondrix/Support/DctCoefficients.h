#ifndef ONDRIX_SUPPORT_DCTCOEFFICIENTS_H
#define ONDRIX_SUPPORT_DCTCOEFFICIENTS_H

#include "ondrix/Support/GuardedFixedQuantization.h"

#include <cmath>
#include <cstdint>
#include <optional>

namespace ondrix {

// The declared angle pi*(2n+1)*k/(2N), with the integer argument reduced
// modulo 4N first. The reduction is exact -- cos(pi*x) has period 2 in x, and
// (2n+1)*k is an exact integer -- and it is what keeps the binary64 evaluation
// accurate enough to be admissible at Q31: unreduced, (2n+1)*k reaches 8001 at
// N = 64, so the angle reaches 196 rad and its 2^-52 relative error becomes
// 9.4e-05 Q31 LSB, past any guard that a coefficient 4.9e-03 LSB from a tie
// could survive. Reduced, the angle stays under 2*pi and the error is
// 7.6e-07 LSB. Every emitted Q15, Q31 and f32 coefficient is unchanged by the
// reduction (all 5456 verified, f32 compared bitwise).
inline double dctAngle(int64_t extent, int64_t k, int64_t n) {
  constexpr double kPi = 3.14159265358979323846264338327950288;
  int64_t reduced = ((2 * n + 1) * k) % (4 * extent);
  return kPi * static_cast<double>(reduced) / (2.0 * static_cast<double>(extent));
}

// Type-II DCT coefficient c[k][n] = q15(cos(pi*(2n+1)*k/(2N))) under the same
// tie-guarded round-half-even quantization as the twiddle tables. This is the
// single table generator shared by every `ondrix.dct` consumer — the tensor
// lowering and the bufferization interface must never derive the table twice.
// A 50-digit sweep of all 5456 coefficients for the supported extents shows a
// worst tie margin of 0.0044 LSB (the 124 saturations are the k = 0 rows, by
// declared convention), so all supported extents are admissible; the guard
// remains the fail-closed backstop.
inline std::optional<int64_t> getDctCoefficientQ15(int64_t extent, int64_t k, int64_t n) {
  std::optional<GuardedQ15Value> quantized = quantizeGuardedQ15(std::cos(dctAngle(extent, k, n)));
  if (!quantized)
    return std::nullopt;
  return quantized->value;
}

// The same coefficient at Q31, under the width-scaled guard. The angle is
// evaluated identically; only the scaling and the admissibility distance
// differ, because the binary64 representation granularity in LSB units grows
// with the format (see kQ31TieGuardLsb). A 50-digit sweep of all 5456
// coefficients shows a worst tie margin of 4.93e-03 LSB, about 40 times the
// guard, so every supported extent is admissible.
inline std::optional<int64_t> getDctCoefficientQ31(int64_t extent, int64_t k, int64_t n) {
  std::optional<GuardedQ31Value> quantized = quantizeGuardedQ31(std::cos(dctAngle(extent, k, n)));
  if (!quantized)
    return std::nullopt;
  return quantized->value;
}

// One entry point for both fixed widths, so a consumer that already carries a
// storage width never re-decides which table it means.
inline std::optional<int64_t> getDctCoefficientFixed(unsigned storageWidth, int64_t extent,
                                                     int64_t k, int64_t n) {
  return storageWidth == 32 ? getDctCoefficientQ31(extent, k, n)
                            : getDctCoefficientQ15(extent, k, n);
}

// The same coefficient for the f32 profile. No tie guard, because the Q15
// guard certifies a quantized table against an independently specified value
// while here the binary32 rounding of this binary64 evaluation IS the declared
// constant. That makes the build's libm part of the declaration; C requires no
// accuracy of `cos`, so the exported bits are pinned in
// test/Conversion/ondrix-to-ondsp/moving_average_dct_f32.mlir.
inline float getDctCoefficientF32(int64_t extent, int64_t k, int64_t n) {
  return static_cast<float>(std::cos(dctAngle(extent, k, n)));
}

// Fail-closed admissibility of the complete coefficient matrix of one static
// extent. A consumer checks this once and may then rely on every individual
// coefficient query succeeding.
inline bool hasAdmissibleDctCoefficients(int64_t extent, unsigned storageWidth = 16) {
  for (int64_t k = 0; k < extent; ++k)
    for (int64_t n = 0; n < extent; ++n)
      if (!getDctCoefficientFixed(storageWidth, extent, k, n))
        return false;
  return true;
}

} // namespace ondrix

#endif // ONDRIX_SUPPORT_DCTCOEFFICIENTS_H

#ifndef ONDRIX_SUPPORT_F32TWIDDLETABLES_H
#define ONDRIX_SUPPORT_F32TWIDDLETABLES_H

#include <cmath>
#include <cstdint>

namespace ondrix {

/// One stage twiddle of the interleaved f32 complex profile.
struct F32Twiddle {
  float real;
  float imaginary;
};

// W(size, index) = exp(-2*pi*j*index/size) forward, its conjugate inverse.
// No tie guard, for the reason stated on getDctCoefficientF32: the Q15 guard
// certifies a quantized table against an independently specified value, while
// here the binary32 rounding of this binary64 evaluation IS the declared
// constant. That makes the build's libm part of the declaration, so the
// exported bits are pinned in test/Conversion/ondrix-to-ondsp/cfft_f32.mlir.
inline F32Twiddle getF32Twiddle(bool forward, int64_t size, int64_t index) {
  constexpr double kTwoPi = 6.28318530717958647692528676655900577;
  double angle = kTwoPi * static_cast<double>(index) / static_cast<double>(size);
  double sine = std::sin(angle);
  return {static_cast<float>(std::cos(angle)), static_cast<float>(forward ? -sine : sine)};
}

} // namespace ondrix

#endif // ONDRIX_SUPPORT_F32TWIDDLETABLES_H

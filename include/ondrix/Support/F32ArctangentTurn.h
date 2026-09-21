#ifndef ONDRIX_SUPPORT_F32ARCTANGENTTURN_H
#define ONDRIX_SUPPORT_F32ARCTANGENTTURN_H

#include "llvm/ADT/ArrayRef.h"

namespace ondrix {

/// Minimax coefficients of `atan(r) / (2*pi) = r * g(r*r)` on `r` in [0, 1],
/// entry `i` weighting `r^(2i)`. They are frozen literals, not an
/// in-compiler evaluation, because a fit has no closed form for the build's
/// libm to evaluate; `scripts/generate-f32-arctangent-turn.py` reproduces
/// them. The octant fold leaves the ratio inside [0, 1], so this interval
/// needs no range reduction and no table.
inline llvm::ArrayRef<float> getF32ArctangentTurnCoefficients() {
  static constexpr float kCoefficients[] = {0x1.45f306p-3f,  -0x1.b2988p-5f,  0x1.04a9cap-5f,
                                            -0x1.725f9cp-6f, 0x1.1578fp-6f,   -0x1.875cf8p-7f,
                                            0x1.bd49e8p-8f,  -0x1.4f342ap-9f, 0x1.db9b5ap-12f};
  return kCoefficients;
}

} // namespace ondrix

#endif // ONDRIX_SUPPORT_F32ARCTANGENTTURN_H

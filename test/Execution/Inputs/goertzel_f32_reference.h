#ifndef ONDRIX_TEST_GOERTZEL_F32_REFERENCE_H
#define ONDRIX_TEST_GOERTZEL_F32_REFERENCE_H

#include <math.h>
#include <stdint.h>

/* The f32 Goertzel reference shared by the dialect-sourced and .ox-sourced
 * object gates. It derives the coefficient itself and runs the declared event
 * graph; it is independent of the lowering apart from that declared order. */

/* Mirrors the lowering's quarter-turn evaluation. This does not gate the
 * snap: the unsnapped 6.1e-17 perturbs no exported bit at these extents, so
 * only the conversion test pins it. */
static float goertzelDoubledCoefficient(int64_t bin, int64_t length) {
  static const double kTwoPi = 6.28318530717958647692528676655900577;
  static const double kQuarterTurns[4] = {1.0, 0.0, -1.0, 0.0};
  double cosine = 4 * bin % length == 0 ? kQuarterTurns[(4 * bin / length) % 4]
                                        : cos(kTwoPi * (double)bin / (double)length);
  return 2.0f * (float)cosine;
}

static float goertzelReference(const float *input, int64_t length, int64_t bin, int fused) {
  const float c2 = goertzelDoubledCoefficient(bin, length);
  float s1 = 0.0f;
  float s2 = 0.0f;
  for (int64_t n = 0; n < length; ++n) {
    const float combined = fused ? fmaf(c2, s1, input[n]) : input[n] + c2 * s1;
    const float next = combined - s2;
    s2 = s1;
    s1 = next;
  }
  const float m = c2 * s1;
  return (s1 * s1 + s2 * s2) - m * s2;
}

#endif /* ONDRIX_TEST_GOERTZEL_F32_REFERENCE_H */

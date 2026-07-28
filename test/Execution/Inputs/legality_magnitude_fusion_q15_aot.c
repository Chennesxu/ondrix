#include <stdint.h>
#include <stdio.h>

extern int16_t magnitude_fused_q15(int32_t bin);
extern int16_t magnitude_requantized_q15(int32_t bin);

/* Exact contract arithmetic for both chains, independent of the compiled
 * code. The witness table shows the real-arithmetic fusion identity
 * failing under explicit Q1.15 boundaries in both directions (fused above
 * chain and chain above fused), collapsing small magnitudes to zero, and
 * agreeing where quantization happens to be benign. */

static int64_t roundHalfEvenShift15(int64_t value) {
  int64_t quotient = value >> 15;
  int64_t remainder = value - (quotient << 15);
  if (remainder > 16384 || (remainder == 16384 && (quotient & 1)))
    ++quotient;
  return quotient;
}

static int16_t saturate16(int64_t value) {
  if (value < INT16_MIN)
    return INT16_MIN;
  if (value > INT16_MAX)
    return INT16_MAX;
  return (int16_t)value;
}

static int16_t isqrtNearestSat(int64_t s) {
  int64_t root = 0;
  for (int bit = 15; bit >= 0; --bit) {
    int64_t candidate = root + ((int64_t)1 << bit);
    if (candidate * candidate <= s)
      root = candidate;
  }
  if (s - root * root > root)
    ++root;
  if (root > 32767)
    root = 32767;
  return (int16_t)root;
}

static int32_t pack(int16_t real, int16_t imaginary) {
  return (int32_t)(((uint32_t)(uint16_t)imaginary << 16) | (uint32_t)(uint16_t)real);
}

static int16_t referenceFused(int16_t real, int16_t imaginary) {
  return isqrtNearestSat((int64_t)real * real + (int64_t)imaginary * imaginary);
}

static int16_t referenceChain(int16_t real, int16_t imaginary) {
  int64_t realQ15 = saturate16(roundHalfEvenShift15((int64_t)real * real));
  int64_t imaginaryQ15 = saturate16(roundHalfEvenShift15((int64_t)imaginary * imaginary));
  return isqrtNearestSat((realQ15 + imaginaryQ15) << 15);
}

struct Witness {
  int16_t real;
  int16_t imaginary;
  int divergent; /* 1 when the two contracts must produce different bits */
};

static const struct Witness kWitnesses[] = {
    {200, 100, 1},      /* fused 224, chain 181 */
    {300, 300, 1},      /* fused 424, chain 443 (chain overshoots) */
    {50, 20, 1},        /* fused 54, chain 0 (information destroyed) */
    {23170, 0, 0},      /* large magnitudes coincide */
    {-32768, -32768, 0} /* both saturate to 32767 */
};

int main(void) {
  int failed = 0;
  for (unsigned i = 0; i < sizeof kWitnesses / sizeof kWitnesses[0]; ++i) {
    int16_t real = kWitnesses[i].real;
    int16_t imaginary = kWitnesses[i].imaginary;
    int32_t bin = pack(real, imaginary);

    int16_t fused = magnitude_fused_q15(bin);
    int16_t chain = magnitude_requantized_q15(bin);
    int16_t fusedExpected = referenceFused(real, imaginary);
    int16_t chainExpected = referenceChain(real, imaginary);

    if (fused != fusedExpected || chain != chainExpected) {
      fprintf(stderr, "(%d,%d): compiled fused=%d chain=%d, expected %d/%d\n", real, imaginary,
              fused, chain, fusedExpected, chainExpected);
      failed = 1;
    }
    if (kWitnesses[i].divergent && fused == chain) {
      fprintf(stderr, "(%d,%d): expected divergence, both produced %d\n", real, imaginary, fused);
      failed = 1;
    }
    if (!kWitnesses[i].divergent && fused != chain) {
      fprintf(stderr, "(%d,%d): expected agreement, got %d vs %d\n", real, imaginary, fused, chain);
      failed = 1;
    }
  }
  return failed;
}

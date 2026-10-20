#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uint64_t state = 0x9e3779b97f4a7c15u;

static int16_t sample16(void) {
  state ^= state << 13;
  state ^= state >> 7;
  state ^= state << 17;
  return (int16_t)(state >> 48);
}

static int64_t saturate(int64_t value, int bits) {
  int64_t top = (INT64_C(1) << (bits - 1)) - 1;
  return value > top ? top : value < -top - 1 ? -top - 1 : value;
}

// round(p / 2^shift) with ties toward +infinity: the floor of p / 2^shift + 1/2.
static int64_t round_up(int64_t p, int shift) { return (p + (INT64_C(1) << (shift - 1))) >> shift; }

static int16_t mult15(int16_t a, int16_t b) {
  return (int16_t)saturate(round_up((int64_t)a * b, 15), 16);
}

static int same(float a, float b) { return memcmp(&a, &b, sizeof(a)) == 0; }

int main(void) {
  int16_t x[40], h[40];
  int32_t wide[40];
  for (unsigned i = 0; i < 40; ++i) {
    x[i] = sample16();
    h[i] = sample16();
    wide[i] = (int32_t)(((uint32_t)(uint16_t)sample16() << 16) | (uint16_t)sample16());
  }
  // The saturating corners: -1 * -1 in the products and runs of +-1 in the sums;
  // +-0.5 times an odd constant is an exact tie.
  x[3] = h[3] = INT16_MIN;
  x[5] = x[7] = 16384;
  x[6] = -16384;
  for (unsigned i = 20; i < 40; ++i)
    x[i] = h[i] = i < 30 ? INT16_MAX : INT16_MIN;
  for (unsigned n = 0; n <= 40; ++n) {
    int64_t acc = 0;
    int16_t y = 0;
    float s = 0.0f;
    for (unsigned i = 0; i < n; ++i) {
      int64_t product = round_up(((int64_t)x[i] * 65536) * ((int64_t)h[i] * 65536), 31);
      acc = saturate(acc + saturate(product, 32), 32);
      y = (int16_t)saturate((int64_t)mult15(y, 29491) + mult15(x[i], 3277), 16);
      s = s + (float)wide[i] * 0x1p-31f;
    }
    if (ondrix_mac(x, n, h, n) != acc || ondrix_pole(x, n) != y || !same(ondrix_mean(wide, n), s)) {
      fprintf(stderr, "length %u\n", n);
      return 1;
    }
  }
  static const int16_t c[8] = {1000, -2000, 3000, -4000, 5000, -6000, 7000, -8000};
  for (unsigned offset = 0; offset <= 32; ++offset) {
    int64_t sum = 0;
    for (unsigned i = 0; i < 8; ++i)
      sum += (int64_t)x[offset + i] * c[i];
    int64_t dot = saturate(round_up(sum, 15), 16);
    if (ondrix_gap(x + offset) != saturate(dot - x[offset], 16)) {
      fprintf(stderr, "gap %u\n", offset);
      return 1;
    }
    int16_t t = x[offset + 2];
    int negated = x[offset + 1] == INT16_MIN ? INT16_MAX : -x[offset + 1];
    if (ondrix_above(x + offset, 2, t) != (x[offset] > t && negated <= 100)) {
      fprintf(stderr, "above %u\n", offset);
      return 1;
    }
  }
  return 0;
}

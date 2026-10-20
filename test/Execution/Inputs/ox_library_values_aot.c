#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uint64_t state = 0x9e3779b97f4a7c15u;

// Signs and exponents spread over 2^-20..2^20 make every summation order observable.
static float sample(void) {
  state ^= state << 13;
  state ^= state >> 7;
  state ^= state << 17;
  float mantissa = 1.0f + (float)(state & 0xffff) / 65536.0f;
  int exponent = (int)((state >> 16) % 41) - 20;
  float value = mantissa;
  for (; exponent > 0; --exponent)
    value *= 2.0f;
  for (; exponent < 0; ++exponent)
    value *= 0.5f;
  return (state >> 40) & 1 ? -value : value;
}

static int same(float a, float b) { return memcmp(&a, &b, sizeof(a)) == 0; }

// The off contract: one ordered fold from the signed-zero identity.
static float dot(const float *x, const float *w, unsigned n) {
  float sum = -0.0f;
  for (unsigned i = 0; i < n; ++i)
    sum = sum + x[i] * w[i];
  return sum;
}

int main(void) {
  float x[12], w[12];
  for (unsigned i = 0; i < 12; ++i) {
    x[i] = sample();
    w[i] = sample();
  }
  for (unsigned length = 0; length <= 12; ++length) {
    float d = dot(x, w, length);
    for (int n = 0; n <= 3; ++n) {
      float s = 0.0f;
      for (int i = 0; i < n; ++i)
        s = s * 0.5f + d;
      if (!same(ondrix_blend(x, w, length, n), s)) {
        fprintf(stderr, "blend %u %d\n", length, n);
        return 1;
      }
    }
    if (ondrix_above(x, w, length, d) || !ondrix_above(x, w, length, d - fabsf(d) - 1.0f)) {
      fprintf(stderr, "above %u\n", length);
      return 1;
    }
  }
  float rms = sqrtf(dot(x, x, 8) / 8.0f);
  if (!same(ondrix_spread(x, w), dot(x, w, 8) - rms * dot(w, w, 8))) {
    fprintf(stderr, "spread\n");
    return 1;
  }
  return 0;
}

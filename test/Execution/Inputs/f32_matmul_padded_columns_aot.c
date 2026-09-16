#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uint32_t next(uint32_t *state) {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

// A finite value in (-8, 8): the corpus is about ordering, not range.
static float draw(uint32_t *state) {
  return (float)((int32_t)next(&state) >> 8) / 16777216.0f * 8.0f;
}

// Bitwise, except that two NaNs agree: their payload lies outside the contract.
static int differs(const char *label, const float *batched, const float *ordered, int count) {
  for (int i = 0; i < count; ++i)
    if (memcmp(&batched[i], &ordered[i], sizeof(float)) != 0 &&
        !(isnan(batched[i]) && isnan(ordered[i]))) {
      fprintf(stderr, "%s[%d]: batched %a, ordered %a\n", label, i, batched[i], ordered[i]);
      return 1;
    }
  return 0;
}

int main(void) {
  uint32_t state = 0x5A17C3E9u;
  float a[4 * 16], b[16 * 3], vec[4 * 3], ord[4 * 3];
  float a6[2 * 4], b6[4 * 6], vec6[2 * 6], ord6[2 * 6];
  for (int trial = 0; trial < 64; ++trial) {
    for (int i = 0; i < 4 * 16; ++i)
      a[i] = draw(&state);
    for (int i = 0; i < 16 * 3; ++i)
      b[i] = draw(&state);
    for (int i = 0; i < 2 * 4; ++i)
      a6[i] = draw(&state);
    for (int i = 0; i < 4 * 6; ++i)
      b6[i] = draw(&state);
    if (trial % 8 == 1) {
      // The surplus lanes read the following row, so an infinity or a NaN
      // there rides a surplus lane for one term and a real lane for the next;
      // both objects must still agree on every stored column.
      b[1 * 3 + 0] = INFINITY;
      b[9 * 3 + 0] = -INFINITY;
      b6[1 * 6 + 0] = INFINITY;
      b6[1 * 6 + 1] = NAN;
    }
    ondrix_f32_matmul_narrow_vec(a, b, vec);
    ondrix_f32_matmul_narrow_ord(a, b, ord);
    if (differs("narrow", vec, ord, 4 * 3))
      return 1;
    ondrix_f32_matmul_six_vec(a6, b6, vec6);
    ondrix_f32_matmul_six_ord(a6, b6, ord6);
    if (differs("six", vec6, ord6, 2 * 6))
      return 1;
  }
  return 0;
}

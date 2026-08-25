#include <stdint.h>
#include <stdio.h>

/* Every extent on the width-4 ladder. The chunked schedule reassociates, which
 * exact-modulo i40 addition makes value-neutral, so the reference is the plain
 * ordered sum taken mod 2^40. */
extern int16_t reduce3(int16_t *, int16_t *, int64_t, int64_t, int64_t, int16_t *, int16_t *,
                       int64_t, int64_t, int64_t);
extern int16_t reduce4(int16_t *, int16_t *, int64_t, int64_t, int64_t, int16_t *, int16_t *,
                       int64_t, int64_t, int64_t);
extern int16_t reduce5(int16_t *, int16_t *, int64_t, int64_t, int64_t, int16_t *, int16_t *,
                       int64_t, int64_t, int64_t);
extern int16_t reduce8(int16_t *, int16_t *, int64_t, int64_t, int64_t, int16_t *, int16_t *,
                       int64_t, int64_t, int64_t);
extern int16_t reduce9(int16_t *, int16_t *, int64_t, int64_t, int64_t, int16_t *, int16_t *,
                       int64_t, int64_t, int64_t);
extern int16_t reduce12(int16_t *, int16_t *, int64_t, int64_t, int64_t, int16_t *, int16_t *,
                        int64_t, int64_t, int64_t);
extern int16_t reduce13(int16_t *, int16_t *, int64_t, int64_t, int64_t, int16_t *, int16_t *,
                        int64_t, int64_t, int64_t);
extern int16_t reduce15(int16_t *, int16_t *, int64_t, int64_t, int64_t, int16_t *, int16_t *,
                        int64_t, int64_t, int64_t);
extern int16_t reduce16(int16_t *, int16_t *, int64_t, int64_t, int64_t, int16_t *, int16_t *,
                        int64_t, int64_t, int64_t);
extern int16_t reduce17(int16_t *, int16_t *, int64_t, int64_t, int64_t, int16_t *, int16_t *,
                        int64_t, int64_t, int64_t);

static int16_t dispatch(int64_t extent, int16_t *lhs, int16_t *rhs) {
  switch (extent) {
  case 3:
    return reduce3(lhs, lhs, 0, 3, 1, rhs, rhs, 0, 3, 1);
  case 4:
    return reduce4(lhs, lhs, 0, 4, 1, rhs, rhs, 0, 4, 1);
  case 5:
    return reduce5(lhs, lhs, 0, 5, 1, rhs, rhs, 0, 5, 1);
  case 8:
    return reduce8(lhs, lhs, 0, 8, 1, rhs, rhs, 0, 8, 1);
  case 9:
    return reduce9(lhs, lhs, 0, 9, 1, rhs, rhs, 0, 9, 1);
  case 12:
    return reduce12(lhs, lhs, 0, 12, 1, rhs, rhs, 0, 12, 1);
  case 13:
    return reduce13(lhs, lhs, 0, 13, 1, rhs, rhs, 0, 13, 1);
  case 15:
    return reduce15(lhs, lhs, 0, 15, 1, rhs, rhs, 0, 15, 1);
  case 16:
    return reduce16(lhs, lhs, 0, 16, 1, rhs, rhs, 0, 16, 1);
  case 17:
    return reduce17(lhs, lhs, 0, 17, 1, rhs, rhs, 0, 17, 1);
  default:
    return 0;
  }
}

static int64_t wrapI40(__int128 value) {
  const unsigned __int128 mask = ((unsigned __int128)1 << 40) - 1;
  const uint64_t bits = (uint64_t)((unsigned __int128)value & mask);
  __int128 widened = bits;
  if (bits & (UINT64_C(1) << 39))
    widened -= (__int128)1 << 40;
  return (int64_t)widened;
}

/* toward_negative at shift 15 with a wrapping i16 destination. */
static int16_t exportQ15(int64_t accumulator) {
  int64_t quotient = accumulator >> 15;
  return (int16_t)(uint16_t)(uint64_t)quotient;
}

static int16_t reference(int64_t extent, const int16_t *lhs, const int16_t *rhs) {
  int64_t accumulator = 0;
  for (int64_t i = 0; i < extent; ++i)
    accumulator = wrapI40((__int128)accumulator + (__int128)lhs[i] * (__int128)rhs[i]);
  return exportQ15(accumulator);
}

static uint32_t nextRandom(uint32_t *state) {
  uint32_t value = *state;
  value ^= value << 13;
  value ^= value >> 17;
  value ^= value << 5;
  *state = value;
  return value;
}

int main(void) {
  static const int64_t kExtents[] = {3, 4, 5, 8, 9, 12, 13, 15, 16, 17};
  const int64_t kExtentCount = (int64_t)(sizeof kExtents / sizeof kExtents[0]);
  int failed = 0;
  for (int64_t e = 0; e < kExtentCount; ++e) {
    const int64_t extent = kExtents[e];
    int16_t lhs[32], rhs[32];
    /* Directed rails first: both full-scale corners and the alternating rail
     * put the widest products at the chunk boundaries the ladder chose. */
    for (int pattern = 0; pattern < 4; ++pattern) {
      for (int64_t i = 0; i < extent; ++i) {
        switch (pattern) {
        case 0:
          lhs[i] = INT16_MIN;
          rhs[i] = INT16_MIN;
          break;
        case 1:
          lhs[i] = INT16_MAX;
          rhs[i] = INT16_MAX;
          break;
        case 2:
          lhs[i] = (i & 1) ? INT16_MAX : INT16_MIN;
          rhs[i] = (i & 1) ? INT16_MIN : INT16_MAX;
          break;
        default:
          lhs[i] = (int16_t)(i + 1);
          rhs[i] = (int16_t) - (i + 1);
          break;
        }
      }
      const int16_t got = dispatch(extent, lhs, rhs);
      const int16_t want = reference(extent, lhs, rhs);
      if (got != want) {
        fprintf(stderr, "extent %lld pattern %d: got %d, expected %d\n", (long long)extent, pattern,
                got, want);
        failed = 1;
      }
    }
    uint32_t state = 0x1F123BB5u ^ (uint32_t)extent;
    for (int trial = 0; trial < 64; ++trial) {
      for (int64_t i = 0; i < extent; ++i) {
        lhs[i] = (int16_t)(nextRandom(&state) >> 16);
        rhs[i] = (int16_t)(nextRandom(&state) >> 16);
      }
      const int16_t got = dispatch(extent, lhs, rhs);
      const int16_t want = reference(extent, lhs, rhs);
      if (got != want) {
        fprintf(stderr, "extent %lld trial %d: got %d, expected %d\n", (long long)extent, trial,
                got, want);
        failed = 1;
      }
    }
  }
  return failed;
}

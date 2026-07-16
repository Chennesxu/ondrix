#include <stdint.h>
#include <stdio.h>

extern int64_t ortumcore_repeat_mac(int16_t lhs, int16_t rhs, int64_t count);
extern int64_t ortumcore_repeat_mac_sub(int16_t lhs, int16_t rhs, int64_t count);

static int64_t update_reference(int64_t accumulator, int16_t lhs, int16_t rhs, int subtract) {
  const __int128 minimum = -((__int128)1 << 39);
  const __int128 maximum = ((__int128)1 << 39) - 1;
  __int128 product = (__int128)lhs * (__int128)rhs;
  __int128 updated = subtract ? (__int128)accumulator - product : (__int128)accumulator + product;
  if (updated < minimum)
    return (int64_t)minimum;
  if (updated > maximum)
    return (int64_t)maximum;
  return (int64_t)updated;
}

static int64_t repeat_reference(int16_t lhs, int16_t rhs, int64_t count, int subtract) {
  int64_t accumulator = 0;
  for (int64_t i = 0; i < count; ++i)
    accumulator = update_reference(accumulator, lhs, rhs, subtract);
  return accumulator;
}

static int check(const char *name, int16_t lhs, int16_t rhs, int64_t count, int subtract) {
  int64_t expected = repeat_reference(lhs, rhs, count, subtract);
  int64_t actual =
      subtract ? ortumcore_repeat_mac_sub(lhs, rhs, count) : ortumcore_repeat_mac(lhs, rhs, count);
  if (actual == expected)
    return 0;
  fprintf(stderr, "%s(%lld): expected %lld, got %lld\n", name, (long long)count,
          (long long)expected, (long long)actual);
  return 1;
}

int main(void) {
  static const int64_t counts[] = {0, 1, 2, 511, 512, 513};
  int failed = 0;
  for (unsigned i = 0; i < sizeof(counts) / sizeof(counts[0]); ++i) {
    failed |= check("target MAC", INT16_MIN, INT16_MIN, counts[i], 0);
    failed |= check("target MAC-sub", INT16_MIN, INT16_MIN, counts[i], 1);
  }
  failed |= check("mixed-sign target MAC", INT16_MIN, INT16_MAX, 19, 0);
  failed |= check("mixed-sign target MAC-sub", INT16_MIN, INT16_MAX, 19, 1);
  return failed;
}

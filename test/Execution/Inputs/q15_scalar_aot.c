#include <stdint.h>
#include <stdio.h>

extern int16_t q15_mul_nearest_even(int16_t lhs, int16_t rhs);
extern int16_t q15_mul_nearest_even_wrap(int16_t lhs, int16_t rhs);
extern int16_t q15_mul_toward_negative(int16_t lhs, int16_t rhs);
extern int16_t q15_mul_toward_zero(int16_t lhs, int16_t rhs);
extern int16_t q15_import_mac_nearest_even(int16_t seed, int16_t lhs, int16_t rhs);
extern int16_t q15_repeat_mac_saturate(int16_t lhs, int16_t rhs, int64_t count);
extern int16_t q15_repeat_mac_wrap(int16_t lhs, int16_t rhs, int64_t count);
extern int16_t q15_repeat_mac_sub_saturate(int16_t lhs, int16_t rhs, int64_t count);
extern int16_t q15_repeat_mac_sub_wrap(int16_t lhs, int16_t rhs, int64_t count);

static int check(const char *name, int16_t lhs, int16_t rhs, int16_t expected) {
  int16_t actual = q15_mul_nearest_even(lhs, rhs);
  if (actual == expected)
    return 0;
  fprintf(stderr, "%s: expected %d, got %d\n", name, expected, actual);
  return 1;
}

typedef int16_t (*repeat_kernel)(int16_t, int16_t, int64_t);

static int64_t update_reference(int64_t accumulator, int16_t lhs, int16_t rhs, int subtract,
                                int saturate) {
  const __int128 minimum = -((__int128)1 << 39);
  const __int128 maximum = ((__int128)1 << 39) - 1;
  __int128 product = (__int128)lhs * (__int128)rhs;
  __int128 updated = subtract ? (__int128)accumulator - product : (__int128)accumulator + product;
  if (saturate) {
    if (updated < minimum)
      return (int64_t)minimum;
    if (updated > maximum)
      return (int64_t)maximum;
    return (int64_t)updated;
  }

  const uint64_t mask = (UINT64_C(1) << 40) - 1;
  uint64_t bits = (uint64_t)updated & mask;
  return (bits & (UINT64_C(1) << 39)) ? (int64_t)(bits - (UINT64_C(1) << 40)) : (int64_t)bits;
}

static uint16_t export_floor_wrap_reference(int64_t accumulator) {
  const int64_t divisor = INT64_C(1) << 15;
  int64_t quotient = accumulator / divisor;
  if (accumulator < 0 && accumulator % divisor != 0)
    --quotient;
  return (uint16_t)quotient;
}

static uint16_t repeat_reference(int16_t lhs, int16_t rhs, int64_t count, int subtract,
                                 int saturate) {
  int64_t accumulator = 0;
  for (int64_t i = 0; i < count; ++i)
    accumulator = update_reference(accumulator, lhs, rhs, subtract, saturate);
  return export_floor_wrap_reference(accumulator);
}

static int check_repeat(const char *name, repeat_kernel kernel, int16_t lhs, int16_t rhs,
                        int64_t count, int subtract, int saturate) {
  uint16_t actual = (uint16_t)kernel(lhs, rhs, count);
  uint16_t expected = repeat_reference(lhs, rhs, count, subtract, saturate);
  if (actual == expected)
    return 0;
  fprintf(stderr, "%s(%d, %d, %lld): expected bits 0x%04x, got 0x%04x\n", name, lhs, rhs,
          (long long)count, expected, actual);
  return 1;
}

static int check_repeated_updates(void) {
  static const int64_t lengths[] = {0, 1, 2, 8, 511, 512, 513};
  int failed = 0;
  for (unsigned i = 0; i < sizeof(lengths) / sizeof(lengths[0]); ++i) {
    int64_t count = lengths[i];
    failed |= check_repeat("mac saturate", q15_repeat_mac_saturate, -32768, -32768, count, 0, 1);
    failed |= check_repeat("mac wrap", q15_repeat_mac_wrap, -32768, -32768, count, 0, 0);
    failed |=
        check_repeat("mac-sub saturate", q15_repeat_mac_sub_saturate, -32768, -32768, count, 1, 1);
    failed |= check_repeat("mac-sub wrap", q15_repeat_mac_sub_wrap, -32768, -32768, count, 1, 0);
  }

  uint32_t state = UINT32_C(0x6d2b79f5);
  for (unsigned i = 0; i < 64; ++i) {
    state = state * UINT32_C(1664525) + UINT32_C(1013904223);
    int16_t lhs = (int16_t)(state >> 16);
    state = state * UINT32_C(1664525) + UINT32_C(1013904223);
    int16_t rhs = (int16_t)(state >> 16);
    state = state * UINT32_C(1664525) + UINT32_C(1013904223);
    int64_t count = state % 520;
    failed |= check_repeat("random mac saturate", q15_repeat_mac_saturate, lhs, rhs, count, 0, 1);
    failed |= check_repeat("random mac wrap", q15_repeat_mac_wrap, lhs, rhs, count, 0, 0);
    failed |=
        check_repeat("random mac-sub saturate", q15_repeat_mac_sub_saturate, lhs, rhs, count, 1, 1);
    failed |= check_repeat("random mac-sub wrap", q15_repeat_mac_sub_wrap, lhs, rhs, count, 1, 0);
  }
  return failed;
}

int main(void) {
  int failed = 0;
  failed |= check("quarter", 16384, 16384, 8192);
  failed |= check("maximum product", 32767, 32767, 32766);
  failed |= check("positive saturation", -32768, -32768, 32767);
  failed |= check("negative product", -32768, 32767, -32767);
  failed |= check("positive even tie", 1, 16384, 0);
  failed |= check("positive odd tie", 3, 16384, 2);
  failed |= check("negative odd tie", -1, 16384, 0);
  failed |= check("negative even tie", -3, 16384, -2);
  failed |= check("nearest below positive tie", 1, 16383, 0);
  failed |= check("nearest above positive tie", 1, 16385, 1);
  failed |= check("nearest below negative tie", -1, 16383, 0);
  failed |= check("nearest above negative tie", -1, 16385, -1);
  if (q15_mul_nearest_even_wrap(-32768, -32768) != -32768) {
    fprintf(stderr, "destination wrap mismatch\n");
    failed = 1;
  }
  if (q15_mul_toward_negative(-1, 16384) != -1 || q15_mul_toward_zero(-1, 16384) != 0) {
    fprintf(stderr, "directed rounding mismatch\n");
    failed = 1;
  }
  if (q15_import_mac_nearest_even(16384, 16384, 16384) != 24576) {
    fprintf(stderr, "accumulator import/update mismatch\n");
    failed = 1;
  }
  failed |= check_repeated_updates();
  return failed;
}

#include <stdint.h>
#include <stdio.h>

extern int16_t q15_mul_nearest_even(int16_t lhs, int16_t rhs);

static int check(const char *name, int16_t lhs, int16_t rhs, int16_t expected) {
  int16_t actual = q15_mul_nearest_even(lhs, rhs);
  if (actual == expected)
    return 0;
  fprintf(stderr, "%s: expected %d, got %d\n", name, expected, actual);
  return 1;
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
  return failed;
}

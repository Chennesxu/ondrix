#include <stdint.h>
#include <stdio.h>

extern int16_t q15_auto_vector_saturate(int16_t *, int16_t *, int64_t, int64_t, int64_t, int16_t *,
                                        int16_t *, int64_t, int64_t, int64_t);

int main(void) {
  int16_t lhs[8] = {0};
  int16_t rhs[7] = {0};
  (void)q15_auto_vector_saturate(lhs, lhs, 0, 8, 1, rhs, rhs, 0, 7, 1);
  fputs("mismatched vectorized memref lengths did not fail\n", stderr);
  return 0;
}

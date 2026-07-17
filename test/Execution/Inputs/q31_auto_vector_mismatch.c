#include <stdint.h>

extern int64_t q31_vector_full_raw_saturate(int32_t, int32_t *, int32_t *, int64_t, int64_t,
                                            int64_t, int32_t *, int32_t *, int64_t, int64_t,
                                            int64_t);

int main(void) {
  int32_t lhs[4] = {0};
  int32_t rhs[4] = {0};
  (void)q31_vector_full_raw_saturate(0, lhs, lhs, 0, 4, 1, rhs, rhs, 0, 3, 1);
  return 0;
}

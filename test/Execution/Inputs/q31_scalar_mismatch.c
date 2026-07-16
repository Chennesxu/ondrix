#include <stdint.h>

extern int32_t q31_reduce_full(int32_t *lhs_allocated, int32_t *lhs_aligned, int64_t lhs_offset,
                               int64_t lhs_size, int64_t lhs_stride, int32_t *rhs_allocated,
                               int32_t *rhs_aligned, int64_t rhs_offset, int64_t rhs_size,
                               int64_t rhs_stride);

int main(void) {
  int32_t lhs[] = {1, 2};
  int32_t rhs[] = {3};
  (void)q31_reduce_full(lhs, lhs, 0, 2, 1, rhs, rhs, 0, 1, 1);
  return 0;
}

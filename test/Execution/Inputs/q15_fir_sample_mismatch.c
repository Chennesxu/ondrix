#include <stdint.h>
#include <stdio.h>

extern int16_t fir_q15_saturate(int16_t *input_allocated, int16_t *input_aligned,
                                int64_t input_offset, int64_t input_size, int64_t input_stride,
                                int16_t *coeffs_allocated, int16_t *coeffs_aligned,
                                int64_t coeffs_offset, int64_t coeffs_size, int64_t coeffs_stride);

int main(void) {
  int16_t input[] = {1, 2};
  int16_t coeffs[] = {3};
  (void)fir_q15_saturate(input, input, 0, 2, 1, coeffs, coeffs, 0, 1, 1);
  fputs("mismatched memref lengths did not fail\n", stderr);
  return 0;
}

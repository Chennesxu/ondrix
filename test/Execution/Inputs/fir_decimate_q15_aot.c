#include "fixed_point_reference.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int16_t *allocated;
  int16_t *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefI16;

extern void _mlir_ciface_fir_decimate_q15(MemRefI16 *result, MemRefI16 *input,
                                          MemRefI16 *coefficients);

static const struct Policy policy = {
    .width = 16,
    .frac = 15,
    .accumulator_width = 40,
    .accumulator_frac = 30,
    .update_overflow = SATURATE,
    .state_rounding = NEAREST_EVEN,
    .state_overflow = SATURATE,
    .output_rounding = NEAREST_EVEN,
    .output_overflow = SATURATE,
};

static void reference(const int16_t input[12], const int16_t coefficients[5], int16_t output[4]) {
  for (int64_t result = 0; result < 4; ++result) {
    int64_t accumulator = 0;
    for (int64_t tap = 0; tap < 5; ++tap)
      accumulator =
          update_reference(accumulator, input[result * 2 + tap], coefficients[tap], &policy);
    output[result] = (int16_t)export_reference(accumulator, policy.output_rounding,
                                               policy.output_overflow, &policy);
  }
}

int main(void) {
  int16_t input[12] = {INT16_MIN, INT16_MAX, 16384, -8192,  4096,  -2048,
                       1024,      -512,      12345, -23456, 30000, -30000};
  int16_t coefficients[5] = {INT16_MAX, -16384, 8192, -4096, 2048};
  const int16_t golden[4] = {INT16_MIN, 21824, 6211, 9173};
  int16_t expected[4];
  reference(input, coefficients, expected);

  MemRefI16 input_ref = {input, input, 0, {12}, {1}};
  MemRefI16 coefficient_ref = {coefficients, coefficients, 0, {5}, {1}};
  MemRefI16 result;
  _mlir_ciface_fir_decimate_q15(&result, &input_ref, &coefficient_ref);

  int failed = result.sizes[0] != 4;
  for (int64_t index = 0; index < 4; ++index) {
    if (expected[index] != golden[index]) {
      fprintf(stderr, "reference[%lld]: got %d, golden %d\n", (long long)index, expected[index],
              golden[index]);
      failed = 1;
    }
    int16_t actual = result.aligned[result.offset + index * result.strides[0]];
    if (actual != expected[index]) {
      fprintf(stderr, "output[%lld]: got %d, expected %d\n", (long long)index, actual,
              expected[index]);
      failed = 1;
    }
  }
  free(result.allocated);
  return failed;
}

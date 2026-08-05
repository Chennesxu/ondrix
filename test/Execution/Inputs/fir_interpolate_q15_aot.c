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

#ifndef FIR_INTERPOLATE_SYMBOL
#define FIR_INTERPOLATE_SYMBOL _mlir_ciface_fir_interpolate_q15
#endif

#ifndef FIR_INTERPOLATE_ACCUMULATOR_WIDTH
#define FIR_INTERPOLATE_ACCUMULATOR_WIDTH 40
#endif

#ifndef FIR_INTERPOLATE_UPDATE_OVERFLOW
#define FIR_INTERPOLATE_UPDATE_OVERFLOW SATURATE
#endif

extern void FIR_INTERPOLATE_SYMBOL(MemRefI16 *result, MemRefI16 *input, MemRefI16 *coefficients);

#ifdef FIR_INTERPOLATE_TIES_POSITIVE_SYMBOL
extern void FIR_INTERPOLATE_TIES_POSITIVE_SYMBOL(MemRefI16 *result, MemRefI16 *input,
                                                 MemRefI16 *coefficients);
#endif

static const struct Policy policy = {
    .width = 16,
    .frac = 15,
    .accumulator_width = FIR_INTERPOLATE_ACCUMULATOR_WIDTH,
    .accumulator_frac = 30,
    .update_overflow = FIR_INTERPOLATE_UPDATE_OVERFLOW,
    .state_rounding = NEAREST_EVEN,
    .state_overflow = SATURATE,
    .output_rounding = NEAREST_EVEN,
    .output_overflow = SATURATE,
};

static void reference(const int16_t input[4], const int16_t coefficients[3], int16_t output[9],
                      enum RoundingMode output_rounding) {
  for (int64_t result = 0; result < 9; ++result) {
    int64_t accumulator = 0;
    for (int64_t tap = 0; tap < 3; ++tap) {
      if (result < tap)
        continue;
      int64_t upsampled_index = result - tap;
      if (upsampled_index % 2 != 0)
        continue;
      int64_t input_index = upsampled_index / 2;
      if (input_index >= 4)
        continue;
      accumulator = update_reference(accumulator, input[input_index], coefficients[tap], &policy);
    }
    output[result] =
        (int16_t)export_reference(accumulator, output_rounding, policy.output_overflow, &policy);
  }
}

int main(void) {
  int16_t input[4] = {INT16_MIN, INT16_MAX, 16384, -8192};
  int16_t coefficients[3] = {INT16_MAX, -16384, 8192};
  const int16_t golden[9] = {-32767, 16384, 24574, -16384, 24575, -8192, -4096, 4096, -2048};
  int16_t expected[9];
  reference(input, coefficients, expected, policy.output_rounding);

  MemRefI16 input_ref = {input, input, 0, {4}, {1}};
  MemRefI16 coefficient_ref = {coefficients, coefficients, 0, {3}, {1}};
  MemRefI16 result;
  FIR_INTERPOLATE_SYMBOL(&result, &input_ref, &coefficient_ref);

  int failed = result.sizes[0] != 9;
  for (int64_t index = 0; index < 9; ++index) {
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

#ifdef FIR_INTERPOLATE_TIES_POSITIVE_SYMBOL
  // Ties-positive export witnesses: taps {+0.5, +0.5, 0} replay each sample at
  // two upsampled positions as 16384*x, so odd samples are exact half ties;
  // x=1 and x=-3 sit on even floor quotients, where the nearest rules diverge
  // at both signs.
  int16_t ntp_input[4] = {1, -3, 0, 0};
  int16_t ntp_coefficients[3] = {16384, 16384, 0};
  const int16_t ntp_golden[9] = {1, 1, -1, -1, 0, 0, 0, 0, 0};
  int16_t ntp_expected[9];
  int16_t ntp_nearest_even[9];
  reference(ntp_input, ntp_coefficients, ntp_expected, NEAREST_TIES_POSITIVE);
  reference(ntp_input, ntp_coefficients, ntp_nearest_even, NEAREST_EVEN);

  MemRefI16 ntp_input_ref = {ntp_input, ntp_input, 0, {4}, {1}};
  MemRefI16 ntp_coefficient_ref = {ntp_coefficients, ntp_coefficients, 0, {3}, {1}};
  MemRefI16 ntp_result;
  FIR_INTERPOLATE_TIES_POSITIVE_SYMBOL(&ntp_result, &ntp_input_ref, &ntp_coefficient_ref);

  int ntp_divergent = 0;
  if (ntp_result.sizes[0] != 9)
    failed = 1;
  for (int64_t index = 0; index < 9; ++index) {
    if (ntp_expected[index] != ntp_golden[index]) {
      fprintf(stderr, "ties-positive reference[%lld]: got %d, golden %d\n", (long long)index,
              ntp_expected[index], ntp_golden[index]);
      failed = 1;
    }
    int16_t actual = ntp_result.aligned[ntp_result.offset + index * ntp_result.strides[0]];
    if (actual != ntp_expected[index]) {
      fprintf(stderr, "ties-positive output[%lld]: got %d, expected %d\n", (long long)index, actual,
              ntp_expected[index]);
      failed = 1;
    }
    if (ntp_expected[index] != ntp_nearest_even[index])
      ++ntp_divergent;
  }
  if (ntp_divergent != 4) {
    fprintf(stderr,
            "ties-positive corpus must diverge from nearest_even on 4 of 9 outputs, "
            "diverged on %d\n",
            ntp_divergent);
    failed = 1;
  }
  free(ntp_result.allocated);
#endif

  return failed;
}

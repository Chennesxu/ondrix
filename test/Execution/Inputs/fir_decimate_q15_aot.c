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

#ifndef FIR_DECIMATE_SYMBOL
#define FIR_DECIMATE_SYMBOL _mlir_ciface_fir_decimate_q15
#endif

#ifndef FIR_DECIMATE_ACCUMULATOR_WIDTH
#define FIR_DECIMATE_ACCUMULATOR_WIDTH 40
#endif

#ifndef FIR_DECIMATE_UPDATE_OVERFLOW
#define FIR_DECIMATE_UPDATE_OVERFLOW SATURATE
#endif

extern void FIR_DECIMATE_SYMBOL(MemRefI16 *result, MemRefI16 *input, MemRefI16 *coefficients);

#ifdef FIR_DECIMATE_TIES_POSITIVE_SYMBOL
extern void FIR_DECIMATE_TIES_POSITIVE_SYMBOL(MemRefI16 *result, MemRefI16 *input,
                                              MemRefI16 *coefficients);
#endif

static const struct Policy policy = {
    .width = 16,
    .frac = 15,
    .accumulator_width = FIR_DECIMATE_ACCUMULATOR_WIDTH,
    .accumulator_frac = 30,
    .update_overflow = FIR_DECIMATE_UPDATE_OVERFLOW,
    .state_rounding = NEAREST_EVEN,
    .state_overflow = SATURATE,
    .output_rounding = NEAREST_EVEN,
    .output_overflow = SATURATE,
};

static void reference(const int16_t input[12], const int16_t coefficients[5], int16_t output[4],
                      enum RoundingMode output_rounding) {
  for (int64_t result = 0; result < 4; ++result) {
    int64_t accumulator = 0;
    for (int64_t tap = 0; tap < 5; ++tap)
      accumulator =
          update_reference(accumulator, input[result * 2 + tap], coefficients[tap], &policy);
    output[result] =
        (int16_t)export_reference(accumulator, output_rounding, policy.output_overflow, &policy);
  }
}

int main(void) {
  int16_t input[12] = {INT16_MIN, INT16_MAX, 16384, -8192,  4096,  -2048,
                       1024,      -512,      12345, -23456, 30000, -30000};
  int16_t coefficients[5] = {INT16_MAX, -16384, 8192, -4096, 2048};
  const int16_t golden[4] = {INT16_MIN, 21824, 6211, 9173};
  int16_t expected[4];
  reference(input, coefficients, expected, policy.output_rounding);

  MemRefI16 input_ref = {input, input, 0, {12}, {1}};
  MemRefI16 coefficient_ref = {coefficients, coefficients, 0, {5}, {1}};
  MemRefI16 result;
  FIR_DECIMATE_SYMBOL(&result, &input_ref, &coefficient_ref);

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

#ifdef FIR_DECIMATE_TIES_POSITIVE_SYMBOL
  // Ties-positive export witnesses: taps +-0.5 make each accumulator
  // 16384*(a-b), so odd window differences are exact half ties; even floor
  // quotients diverge from nearest_even at both signs, and difference 65533
  // lands the ties-positive increment exactly on +32767.
  int16_t ntp_input[12] = {1, 0, 0, 3, 32767, -32766, 0, 0, 0, 0, 0, 0};
  int16_t ntp_coefficients[5] = {16384, -16384, 0, 0, 0};
  const int16_t ntp_golden[4] = {1, -1, 32767, 0};
  int16_t ntp_expected[4];
  int16_t ntp_nearest_even[4];
  reference(ntp_input, ntp_coefficients, ntp_expected, NEAREST_TIES_POSITIVE);
  reference(ntp_input, ntp_coefficients, ntp_nearest_even, NEAREST_EVEN);

  MemRefI16 ntp_input_ref = {ntp_input, ntp_input, 0, {12}, {1}};
  MemRefI16 ntp_coefficient_ref = {ntp_coefficients, ntp_coefficients, 0, {5}, {1}};
  MemRefI16 ntp_result;
  FIR_DECIMATE_TIES_POSITIVE_SYMBOL(&ntp_result, &ntp_input_ref, &ntp_coefficient_ref);

  int ntp_divergent = 0;
  for (int64_t index = 0; index < 4; ++index) {
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
  if (ntp_divergent != 3) {
    fprintf(stderr,
            "ties-positive corpus must diverge from nearest_even on 3 of 4 outputs, "
            "diverged on %d\n",
            ntp_divergent);
    failed = 1;
  }
  free(ntp_result.allocated);
#endif

  return failed;
}

#include <limits.h>
#include <stdint.h>
#include <stdio.h>

typedef int16_t (*q15_kernel_t)(int16_t *, int16_t *, int64_t, int64_t, int64_t);
typedef int32_t (*q31_kernel_t)(int32_t *, int32_t *, int64_t, int64_t, int64_t);

extern int16_t q15_proven_vector(int16_t *, int16_t *, int64_t, int64_t, int64_t);
extern int16_t q15_scalar_reference(int16_t *, int16_t *, int64_t, int64_t, int64_t);
extern int16_t q15_proven_offset(int16_t *, int16_t *, int64_t, int64_t, int64_t);
extern int32_t q31_proven_vector(int32_t *, int32_t *, int64_t, int64_t, int64_t);
extern int32_t q31_scalar_reference(int32_t *, int32_t *, int64_t, int64_t, int64_t);

static const int16_t q15_coefficients[17] = {1,   -2, 3,   -4, 5,   -6, 7,   -8, 9,
                                             -10, 11, -12, 13, -14, 15, -16, 17};
static const int32_t q31_coefficients[5] = {1, -2, 3, -4, 5};

static int64_t saturate_signed(__int128 value, unsigned width) {
  const __int128 minimum = -((__int128)1 << (width - 1));
  const __int128 maximum = ((__int128)1 << (width - 1)) - 1;
  if (value < minimum)
    return (int64_t)minimum;
  if (value > maximum)
    return (int64_t)maximum;
  return (int64_t)value;
}

static int64_t floor_shift(int64_t value, unsigned shift) {
  const int64_t divisor = INT64_C(1) << shift;
  int64_t quotient = value / divisor;
  if (value < 0 && value % divisor != 0)
    --quotient;
  return quotient;
}

static uint16_t reference_q15(const int16_t *input) {
  int64_t accumulator = 0;
  for (unsigned index = 0; index < 17; ++index)
    accumulator = saturate_signed(
        (__int128)accumulator + (__int128)input[index] * q15_coefficients[index], 40);
  return (uint16_t)floor_shift(accumulator, 15);
}

static uint32_t reference_q31(const int32_t *input) {
  int64_t accumulator = 0;
  for (unsigned index = 0; index < 5; ++index)
    accumulator = saturate_signed(
        (__int128)accumulator + (__int128)input[index] * q31_coefficients[index], 64);
  return (uint32_t)floor_shift(accumulator, 31);
}

static int check_q15(int16_t *input) {
  uint16_t expected = reference_q15(input);
  uint16_t vector = (uint16_t)q15_proven_vector(input, input, 0, 17, 1);
  uint16_t scalar = (uint16_t)q15_scalar_reference(input, input, 0, 17, 1);
  if (vector == expected && scalar == expected)
    return 0;
  fprintf(stderr,
          "Q15 constant saturating reduction: expected 0x%04x, vector 0x%04x, scalar 0x%04x\n",
          expected, vector, scalar);
  return 1;
}

static int check_q31(int32_t *input) {
  uint32_t expected = reference_q31(input);
  uint32_t vector = (uint32_t)q31_proven_vector(input, input, 0, 5, 1);
  uint32_t scalar = (uint32_t)q31_scalar_reference(input, input, 0, 5, 1);
  if (vector == expected && scalar == expected)
    return 0;
  fprintf(stderr,
          "Q31 constant saturating reduction: expected 0x%08x, vector 0x%08x, scalar 0x%08x\n",
          expected, vector, scalar);
  return 1;
}

static int check_q15_offset(int16_t *storage) {
  uint16_t expected = reference_q15(storage + 1);
  uint16_t vector = (uint16_t)q15_proven_offset(storage, storage, 1, 17, 1);
  uint16_t scalar = (uint16_t)q15_scalar_reference(storage, storage, 1, 17, 1);
  if (vector == expected && scalar == expected)
    return 0;
  fprintf(stderr, "Q15 offset reduction: expected 0x%04x, vector 0x%04x, scalar 0x%04x\n", expected,
          vector, scalar);
  return 1;
}

int main(void) {
  int16_t q15_inputs[][17] = {
      {INT16_MIN, INT16_MAX, -1, 1, 12345, -23456, 17, -19, 23, -29, 31, -37, 41, -43, 47, -53, 59},
      {1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 17},
  };
  int32_t q31_inputs[][5] = {
      {INT32_MIN, INT32_MAX, -1, 1, 123456789},
      {1, -2, 3, -4, 5},
  };
  int16_t q15_offset_input[18] = {
      123, INT16_MIN, INT16_MAX, -1,  1,  12345, -23456, 17,  -19,
      23,  -29,       31,        -37, 41, -43,   47,     -53, 59,
  };

  int failed = 0;
  for (unsigned index = 0; index < sizeof(q15_inputs) / sizeof(q15_inputs[0]); ++index)
    failed |= check_q15(q15_inputs[index]);
  for (unsigned index = 0; index < sizeof(q31_inputs) / sizeof(q31_inputs[0]); ++index)
    failed |= check_q31(q31_inputs[index]);
  failed |= check_q15_offset(q15_offset_input);
  return failed;
}

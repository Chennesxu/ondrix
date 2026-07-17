#include <limits.h>
#include <stdint.h>
#include <stdio.h>

extern int16_t sparse_q15_saturate(int16_t *allocated, int16_t *aligned, int64_t offset,
                                   int64_t size, int64_t stride);
extern int16_t symmetric_q15_wrap(int16_t *allocated, int16_t *aligned, int64_t offset,
                                  int64_t size, int64_t stride);
extern int16_t symmetric_q15_saturate(int16_t *allocated, int16_t *aligned, int64_t offset,
                                      int64_t size, int64_t stride);
extern int16_t symmetric_q15_strided_wrap(int16_t *allocated, int16_t *aligned, int64_t offset,
                                          int64_t size, int64_t stride);
extern int32_t symmetric_q31_wrap(int32_t *allocated, int32_t *aligned, int64_t offset,
                                  int64_t size, int64_t stride);
extern int32_t symmetric_q31_saturate(int32_t *allocated, int32_t *aligned, int64_t offset,
                                      int64_t size, int64_t stride);

static const int16_t sparse_q15_coefficients[] = {32767, 0, INT16_MIN, 0, 12345};
static const int16_t symmetric_q15_coefficients[] = {32767, INT16_MIN, 12345, INT16_MIN, 32767};
static const int32_t symmetric_q31_coefficients[] = {INT32_MAX, INT32_MIN, INT32_MIN, INT32_MAX};
static const int32_t symmetric_q31_safe_coefficients[] = {1, 2, 2, 1};

static int64_t wrap_signed(__int128 value, unsigned width) {
  const unsigned __int128 mask = ((unsigned __int128)1 << width) - 1;
  const uint64_t bits = (uint64_t)((unsigned __int128)value & mask);
  __int128 signed_value = bits;
  if (bits & (UINT64_C(1) << (width - 1)))
    signed_value -= (__int128)1 << width;
  return (int64_t)signed_value;
}

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

static uint16_t reference_q15(const int16_t *input, const int16_t *coefficients, int saturate) {
  int64_t accumulator = 0;
  for (unsigned index = 0; index < 5; ++index) {
    __int128 update = (__int128)accumulator + (__int128)input[index] * coefficients[index];
    accumulator = saturate ? saturate_signed(update, 40) : wrap_signed(update, 40);
  }
  return (uint16_t)floor_shift(accumulator, 15);
}

static uint32_t reference_q31(const int32_t *input, const int32_t *coefficients, int saturate) {
  int64_t accumulator = 0;
  for (unsigned index = 0; index < 4; ++index) {
    __int128 update = (__int128)accumulator + (__int128)input[index] * coefficients[index];
    accumulator = saturate ? saturate_signed(update, 64) : wrap_signed(update, 64);
  }
  return (uint32_t)floor_shift(accumulator, 31);
}

static int check_q15(const char *name,
                     int16_t (*kernel)(int16_t *, int16_t *, int64_t, int64_t, int64_t),
                     int16_t *input, const int16_t *coefficients, int saturate) {
  uint16_t actual = (uint16_t)kernel(input, input, 0, 5, 1);
  uint16_t expected = reference_q15(input, coefficients, saturate);
  if (actual == expected)
    return 0;
  fprintf(stderr, "%s: expected 0x%04x, got 0x%04x\n", name, expected, actual);
  return 1;
}

static int check_q31(const char *name,
                     int32_t (*kernel)(int32_t *, int32_t *, int64_t, int64_t, int64_t),
                     int32_t *input, const int32_t *coefficients, int saturate) {
  uint32_t actual = (uint32_t)kernel(input, input, 0, 4, 1);
  uint32_t expected = reference_q31(input, coefficients, saturate);
  if (actual == expected)
    return 0;
  fprintf(stderr, "%s: expected 0x%08x, got 0x%08x\n", name, expected, actual);
  return 1;
}

static int check_strided_q15(void) {
  int16_t storage[] = {111, INT16_MIN, 222, INT16_MAX, 333, -1, 444, 1, 555, 12345, 666};
  int16_t logical[5];
  for (unsigned index = 0; index < 5; ++index)
    logical[index] = storage[1 + index * 2];

  uint16_t actual = (uint16_t)symmetric_q15_strided_wrap(storage, storage, 1, 5, 2);
  uint16_t expected = reference_q15(logical, symmetric_q15_coefficients, 0);
  if (actual == expected)
    return 0;
  fprintf(stderr, "strided symmetric Q15: expected 0x%04x, got 0x%04x\n", expected, actual);
  return 1;
}

int main(void) {
  int16_t q15_inputs[][5] = {
      {INT16_MIN, INT16_MAX, -1, 1, 12345},
      {INT16_MAX, INT16_MIN, 23456, -23456, INT16_MIN},
      {INT16_MIN, INT16_MIN, 0, INT16_MIN, INT16_MIN},
      {1, -2, 3, -4, 5},
  };
  int32_t q31_inputs[][4] = {
      {INT32_MIN, INT32_MAX, -1, 1},
      {INT32_MAX, INT32_MIN, INT32_MIN, INT32_MAX},
      {123456789, -987654321, 135791113, -24681012},
  };

  int failed = 0;
  for (unsigned index = 0; index < sizeof(q15_inputs) / sizeof(q15_inputs[0]); ++index) {
    failed |=
        check_q15("sparse Q15", sparse_q15_saturate, q15_inputs[index], sparse_q15_coefficients, 1);
    failed |= check_q15("symmetric Q15", symmetric_q15_wrap, q15_inputs[index],
                        symmetric_q15_coefficients, 0);
    failed |= check_q15("saturating symmetric Q15", symmetric_q15_saturate, q15_inputs[index],
                        symmetric_q15_coefficients, 1);
  }
  for (unsigned index = 0; index < sizeof(q31_inputs) / sizeof(q31_inputs[0]); ++index) {
    failed |= check_q31("symmetric Q31", symmetric_q31_wrap, q31_inputs[index],
                        symmetric_q31_coefficients, 0);
    failed |= check_q31("saturating symmetric Q31", symmetric_q31_saturate, q31_inputs[index],
                        symmetric_q31_safe_coefficients, 1);
  }
  failed |= check_strided_q15();
  return failed;
}

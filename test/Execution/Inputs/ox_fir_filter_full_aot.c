#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DECLARE_MEMREF(NAME, TYPE)                                                                 \
  typedef struct {                                                                                 \
    TYPE *allocated;                                                                               \
    TYPE *aligned;                                                                                 \
    int64_t offset;                                                                                \
    int64_t sizes[1];                                                                              \
    int64_t strides[1];                                                                            \
  } NAME

DECLARE_MEMREF(MemRefI16, int16_t);
DECLARE_MEMREF(MemRefI32, int32_t);
DECLARE_MEMREF(MemRefF32, float);

extern void _mlir_ciface_q15_fir_filter_full(MemRefI16 *, MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_q31_fir_filter_full(MemRefI32 *, MemRefI32 *, MemRefI32 *);
extern void _mlir_ciface_f32_fir_filter_full(MemRefF32 *, MemRefF32 *, MemRefF32 *);

#define MAKE_MEMREF(DATA, COUNT)                                                                   \
  {                                                                                                \
    DATA, DATA, 0, {COUNT}, { 1 }                                                                  \
  }

static int64_t clamp_i40(__int128 value) {
  const __int128 minimum = -((__int128)1 << 39);
  const __int128 maximum = ((__int128)1 << 39) - 1;
  if (value < minimum)
    return (int64_t)minimum;
  if (value > maximum)
    return (int64_t)maximum;
  return (int64_t)value;
}

static int64_t wrap_i64(__int128 value) {
  const __int128 modulus = (__int128)1 << 64;
  __int128 bits = value % modulus;
  if (bits < 0)
    bits += modulus;
  if (bits >= ((__int128)1 << 63))
    bits -= modulus;
  return (int64_t)bits;
}

static int32_t wrap_i32(__int128 value) {
  const __int128 modulus = (__int128)1 << 32;
  __int128 bits = value % modulus;
  if (bits < 0)
    bits += modulus;
  if (bits >= ((__int128)1 << 31))
    bits -= modulus;
  return (int32_t)bits;
}

static __int128 round_nearest_even(__int128 value, unsigned shift) {
  const __int128 divisor = (__int128)1 << shift;
  __int128 quotient = value / divisor;
  __int128 remainder = value % divisor;
  if (remainder < 0) {
    remainder += divisor;
    --quotient;
  }
  const __int128 half = divisor >> 1;
  if (remainder > half || (remainder == half && (quotient & 1)))
    ++quotient;
  return quotient;
}

static int16_t q15_reference(const int16_t *input, int64_t inputCount, const int16_t *coefficients,
                             int64_t coefficientCount, int64_t output) {
  int64_t accumulator = 0;
  for (int64_t tap = 0; tap < coefficientCount; ++tap) {
    int64_t inputIndex = output + tap - (coefficientCount - 1);
    if (inputIndex < 0 || inputIndex >= inputCount)
      continue;
    accumulator =
        clamp_i40((__int128)accumulator + (__int128)input[inputIndex] * coefficients[tap]);
  }
  __int128 value = round_nearest_even(accumulator, 15);
  if (value < INT16_MIN)
    return INT16_MIN;
  if (value > INT16_MAX)
    return INT16_MAX;
  return (int16_t)value;
}

static int32_t q31_reference(const int32_t *input, int64_t inputCount, const int32_t *coefficients,
                             int64_t coefficientCount, int64_t output) {
  int64_t accumulator = 0;
  for (int64_t tap = 0; tap < coefficientCount; ++tap) {
    int64_t inputIndex = output + tap - (coefficientCount - 1);
    if (inputIndex < 0 || inputIndex >= inputCount)
      continue;
    accumulator = wrap_i64((__int128)accumulator + (__int128)input[inputIndex] * coefficients[tap]);
  }
  return wrap_i32((__int128)accumulator / ((__int128)1 << 31));
}

static uint32_t f32_bits(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

int main(void) {
  int failed = 0;

  int16_t q15Input[] = {32767, -32768, 4096, -8192, 16384, 1};
  int16_t q15Coefficients[] = {-32768, 16384, 8192};
  MemRefI16 q15InputRef = MAKE_MEMREF(q15Input, 6);
  MemRefI16 q15CoefficientRef = MAKE_MEMREF(q15Coefficients, 3);
  MemRefI16 q15Output;
  _mlir_ciface_q15_fir_filter_full(&q15Output, &q15InputRef, &q15CoefficientRef);
  if (q15Output.sizes[0] != 8)
    failed = 1;
  for (int64_t i = 0; i < q15Output.sizes[0]; ++i) {
    int16_t expected = q15_reference(q15Input, 6, q15Coefficients, 3, i);
    int16_t actual = q15Output.aligned[q15Output.offset + i * q15Output.strides[0]];
    if (actual != expected) {
      fprintf(stderr, "Q15 output %lld: got %d, expected %d\n", (long long)i, actual, expected);
      failed = 1;
    }
  }
  free(q15Output.allocated);

  int32_t q31Input[] = {INT32_MIN, INT32_MAX, -123456789, 987654321};
  int32_t q31Coefficients[] = {INT32_MIN, INT32_MAX, INT32_MIN};
  MemRefI32 q31InputRef = MAKE_MEMREF(q31Input, 4);
  MemRefI32 q31CoefficientRef = MAKE_MEMREF(q31Coefficients, 3);
  MemRefI32 q31Output;
  _mlir_ciface_q31_fir_filter_full(&q31Output, &q31InputRef, &q31CoefficientRef);
  if (q31Output.sizes[0] != 6)
    failed = 1;
  for (int64_t i = 0; i < q31Output.sizes[0]; ++i) {
    int32_t expected = q31_reference(q31Input, 4, q31Coefficients, 3, i);
    int32_t actual = q31Output.aligned[q31Output.offset + i * q31Output.strides[0]];
    if (actual != expected) {
      fprintf(stderr, "Q31 output %lld: got %d, expected %d\n", (long long)i, actual, expected);
      failed = 1;
    }
  }
  free(q31Output.allocated);

  float f32Input[] = {INFINITY, -3.25f, NAN, -0.0f, 0.5f};
  float f32Coefficients[] = {INFINITY, NAN, 0.25f};
  MemRefF32 f32InputRef = MAKE_MEMREF(f32Input, 5);
  MemRefF32 f32CoefficientRef = MAKE_MEMREF(f32Coefficients, 3);
  MemRefF32 f32Output;
  _mlir_ciface_f32_fir_filter_full(&f32Output, &f32InputRef, &f32CoefficientRef);
  if (f32Output.sizes[0] != 7)
    failed = 1;
  for (int64_t i = 0; i < f32Output.sizes[0]; ++i) {
    float expected = 0.0f;
    for (int64_t tap = 0; tap < 3; ++tap) {
      int64_t inputIndex = i + tap - 2;
      if (inputIndex >= 0 && inputIndex < 5)
        expected = fmaf(f32Input[inputIndex], f32Coefficients[tap], expected);
    }
    float actual = f32Output.aligned[f32Output.offset + i * f32Output.strides[0]];
    if (!(isnan(actual) && isnan(expected)) && f32_bits(actual) != f32_bits(expected)) {
      fprintf(stderr, "f32 output %lld: got 0x%08x, expected 0x%08x\n", (long long)i,
              f32_bits(actual), f32_bits(expected));
      failed = 1;
    }
  }
  free(f32Output.allocated);

  return failed;
}

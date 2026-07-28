#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int32_t *allocated;
  int32_t *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefI32;

extern void _mlir_ciface_q15_cfft8(MemRefI32 *result, MemRefI32 *input);

struct Complex {
  int16_t real;
  int16_t imaginary;
};

static int16_t requantize(__int128 value, unsigned shift) {
  const __int128 divisor = (__int128)1 << shift;
  __int128 quotient = value / divisor;
  __int128 remainder = value % divisor;
  if (remainder < 0) {
    --quotient;
    remainder += divisor;
  }
  const __int128 half = divisor >> 1;
  if (remainder > half || (remainder == half && (quotient & 1)))
    ++quotient;
  if (quotient < INT16_MIN)
    return INT16_MIN;
  if (quotient > INT16_MAX)
    return INT16_MAX;
  return (int16_t)quotient;
}

static int16_t decodeSigned16(uint16_t bits) {
  int32_t value = bits;
  if ((bits & UINT16_C(0x8000)) != 0)
    value -= INT32_C(65536);
  return (int16_t)value;
}

static int32_t toSigned32(uint32_t bits) {
  return bits <= (uint32_t)INT32_MAX ? (int32_t)bits
                                     : (int32_t)((int64_t)bits - (INT64_C(1) << 32));
}

static int32_t pack(struct Complex value) {
  return toSigned32(((uint32_t)(uint16_t)value.imaginary << 16) | (uint32_t)(uint16_t)value.real);
}

static struct Complex unpack(int32_t value) {
  return (struct Complex){decodeSigned16((uint16_t)value),
                          decodeSigned16((uint16_t)((uint32_t)value >> 16))};
}

static void butterfly(int32_t aBits, int32_t bBits, int32_t twiddleBits, int32_t *out0,
                      int32_t *out1) {
  struct Complex a = unpack(aBits);
  struct Complex b = unpack(bBits);
  struct Complex w = unpack(twiddleBits);
  int16_t tr = requantize((__int128)b.real * w.real - (__int128)b.imaginary * w.imaginary, 15);
  int16_t ti = requantize((__int128)b.real * w.imaginary + (__int128)b.imaginary * w.real, 15);
  *out0 = pack((struct Complex){
      requantize((__int128)a.real + tr, 1),
      requantize((__int128)a.imaginary + ti, 1),
  });
  *out1 = pack((struct Complex){
      requantize((__int128)a.real - tr, 1),
      requantize((__int128)a.imaginary - ti, 1),
  });
}

static void cfft4Strided(const int32_t input[8], unsigned offset, int32_t output[4]) {
  const int32_t one = pack((struct Complex){INT16_MAX, 0});
  const int32_t minusJ = pack((struct Complex){0, INT16_MIN});
  int32_t s0, s1, s2, s3;
  butterfly(input[offset], input[offset + 4], one, &s0, &s2);
  butterfly(input[offset + 2], input[offset + 6], one, &s1, &s3);
  butterfly(s0, s1, one, &output[0], &output[2]);
  butterfly(s2, s3, minusJ, &output[1], &output[3]);
}

static void cfft8(const int32_t input[8], int32_t output[8]) {
  static const struct Complex twiddleValues[4] = {
      {INT16_MAX, 0},
      {23170, -23170},
      {0, INT16_MIN},
      {-23170, -23170},
  };
  int32_t even[4];
  int32_t odd[4];
  cfft4Strided(input, 0, even);
  cfft4Strided(input, 1, odd);
  for (unsigned k = 0; k < 4; ++k)
    butterfly(even[k], odd[k], pack(twiddleValues[k]), &output[k], &output[k + 4]);
}

struct Case {
  const char *name;
  struct Complex input[8];
};

int main(void) {
  static const struct Case cases[] = {
      {"zero", {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}},
      {"real-impulse", {{INT16_MAX, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}},
      {"mixed",
       {{12000, -7000},
        {-16000, 23000},
        {INT16_MIN, 5000},
        {9000, INT16_MAX},
        {-4000, 17000},
        {30000, -31000},
        {-22000, -6000},
        {14000, 8000}}},
      {"extrema",
       {{INT16_MIN, INT16_MIN},
        {INT16_MAX, INT16_MAX},
        {INT16_MIN, INT16_MAX},
        {INT16_MAX, INT16_MIN},
        {INT16_MIN, 0},
        {0, INT16_MIN},
        {INT16_MAX, 0},
        {0, INT16_MAX}}},
  };

  for (unsigned caseIndex = 0; caseIndex < sizeof(cases) / sizeof(cases[0]); ++caseIndex) {
    int32_t input[8];
    int32_t expected[8];
    for (unsigned i = 0; i < 8; ++i)
      input[i] = pack(cases[caseIndex].input[i]);
    cfft8(input, expected);

    MemRefI32 inputRef = {input, input, 0, {8}, {1}};
    MemRefI32 output;
    _mlir_ciface_q15_cfft8(&output, &inputRef);
    if (output.sizes[0] != 8) {
      fprintf(stderr, "%s: expected output extent 8, got %" PRId64 "\n", cases[caseIndex].name,
              output.sizes[0]);
      free(output.allocated);
      return 1;
    }
    for (unsigned i = 0; i < 8; ++i) {
      int32_t actual = output.aligned[output.offset + i * output.strides[0]];
      if (actual != expected[i]) {
        fprintf(stderr, "%s[%u]: expected %08" PRIx32 ", got %08" PRIx32 "\n",
                cases[caseIndex].name, i, (uint32_t)expected[i], (uint32_t)actual);
        free(output.allocated);
        return 1;
      }
    }
    free(output.allocated);
  }
  return 0;
}

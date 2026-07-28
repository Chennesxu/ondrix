#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>

extern int32_t cfft4_q15_value(int32_t x0, int32_t x1, int32_t x2, int32_t x3, int64_t index);

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
  struct Complex first = {
      requantize((__int128)a.real + tr, 1),
      requantize((__int128)a.imaginary + ti, 1),
  };
  struct Complex second = {
      requantize((__int128)a.real - tr, 1),
      requantize((__int128)a.imaginary - ti, 1),
  };
  *out0 = pack(first);
  *out1 = pack(second);
}

static void cfft4(const int32_t input[4], int32_t output[4]) {
  const int32_t one = pack((struct Complex){INT16_MAX, 0});
  const int32_t minusJ = pack((struct Complex){0, INT16_MIN});
  int32_t s0, s1, s2, s3;
  butterfly(input[0], input[2], one, &s0, &s2);
  butterfly(input[1], input[3], one, &s1, &s3);
  butterfly(s0, s1, one, &output[0], &output[2]);
  butterfly(s2, s3, minusJ, &output[1], &output[3]);
}

struct Case {
  const char *name;
  struct Complex input[4];
};

int main(void) {
  static const struct Case cases[] = {
      {"zero", {{0, 0}, {0, 0}, {0, 0}, {0, 0}}},
      {"real-impulse", {{INT16_MAX, 0}, {0, 0}, {0, 0}, {0, 0}}},
      {"imag-impulse", {{0, INT16_MIN}, {0, 0}, {0, 0}, {0, 0}}},
      {"mixed", {{12000, -7000}, {-16000, 23000}, {INT16_MIN, 5000}, {9000, INT16_MAX}}},
      {"extrema",
       {{INT16_MIN, INT16_MIN},
        {INT16_MAX, INT16_MAX},
        {INT16_MIN, INT16_MAX},
        {INT16_MAX, INT16_MIN}}},
      {"rounding", {{1, -1}, {3, -3}, {5, -5}, {7, -7}}},
  };

  for (unsigned caseIndex = 0; caseIndex < sizeof(cases) / sizeof(cases[0]); ++caseIndex) {
    int32_t input[4];
    int32_t expected[4];
    for (unsigned i = 0; i < 4; ++i)
      input[i] = pack(cases[caseIndex].input[i]);
    cfft4(input, expected);
    for (unsigned i = 0; i < 4; ++i) {
      int32_t actual = cfft4_q15_value(input[0], input[1], input[2], input[3], (int64_t)i);
      if (actual != expected[i]) {
        fprintf(stderr, "%s[%u]: expected %08" PRIx32 ", got %08" PRIx32 "\n",
                cases[caseIndex].name, i, (uint32_t)expected[i], (uint32_t)actual);
        return 1;
      }
    }
  }
  return 0;
}

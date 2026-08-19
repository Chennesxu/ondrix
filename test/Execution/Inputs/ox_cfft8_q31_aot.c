/* Independent reference for the .ox packed-Q31 CFFT8 binding: the re-frozen
 * raw-high equation with per-term floors, written as a recursive schedule
 * against the frozen Q31 twiddle words. */

#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int64_t *allocated;
  int64_t *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefI64;

extern void _mlir_ciface_q31_cfft8(MemRefI64 *result, MemRefI64 *input);

struct Complex {
  int32_t real;
  int32_t imaginary;
};

static int32_t saturate32(__int128 value) {
  if (value < INT32_MIN)
    return INT32_MIN;
  if (value > INT32_MAX)
    return INT32_MAX;
  return (int32_t)value;
}

static __int128 floorShift(__int128 value, unsigned shift) {
  const __int128 divisor = (__int128)1 << shift;
  __int128 quotient = value / divisor;
  if (value % divisor != 0 && value < 0)
    --quotient;
  return quotient;
}

static int32_t decodeSigned32(uint32_t bits) {
  int64_t value = bits;
  if ((bits & UINT32_C(0x80000000)) != 0)
    value -= INT64_C(4294967296);
  return (int32_t)value;
}

static int64_t toSigned64(uint64_t bits) {
  return bits < UINT64_C(0x8000000000000000)
             ? (int64_t)bits
             : (int64_t)(__int128)((__int128)bits - ((__int128)1 << 64));
}

static int64_t pack(struct Complex value) {
  return toSigned64(((uint64_t)(uint32_t)value.imaginary << 32) | (uint64_t)(uint32_t)value.real);
}

static struct Complex unpack(int64_t value) {
  return (struct Complex){decodeSigned32((uint32_t)value),
                          decodeSigned32((uint32_t)((uint64_t)value >> 32))};
}

/* Each cross term floors at the storage width on its own; the combine is
 * exact, doubled with saturation, and each stage output floors by one. */
static void butterfly(int64_t aBits, int64_t bBits, int64_t twiddleBits, int64_t *out0,
                      int64_t *out1) {
  struct Complex a = unpack(aBits);
  struct Complex b = unpack(bBits);
  struct Complex w = unpack(twiddleBits);
  int64_t termReal = (int64_t)floorShift((__int128)b.real * w.real, 32) -
                     (int64_t)floorShift((__int128)b.imaginary * w.imaginary, 32);
  int64_t termImaginary = (int64_t)floorShift((__int128)b.real * w.imaginary, 32) +
                          (int64_t)floorShift((__int128)b.imaginary * w.real, 32);
  int32_t tr = saturate32((__int128)termReal * 2);
  int32_t ti = saturate32((__int128)termImaginary * 2);
  *out0 = pack((struct Complex){
      saturate32(floorShift((__int128)a.real + tr, 1)),
      saturate32(floorShift((__int128)a.imaginary + ti, 1)),
  });
  *out1 = pack((struct Complex){
      saturate32(floorShift((__int128)a.real - tr, 1)),
      saturate32(floorShift((__int128)a.imaginary - ti, 1)),
  });
}

static void cfft4Strided(const int64_t input[8], unsigned offset, int64_t output[4]) {
  const int64_t one = pack((struct Complex){INT32_MAX, 0});
  const int64_t minusJ = pack((struct Complex){0, INT32_MIN});
  int64_t s0, s1, s2, s3;
  butterfly(input[offset], input[offset + 4], one, &s0, &s2);
  butterfly(input[offset + 2], input[offset + 6], one, &s1, &s3);
  butterfly(s0, s1, one, &output[0], &output[2]);
  butterfly(s2, s3, minusJ, &output[1], &output[3]);
}

static void cfft8(const int64_t input[8], int64_t output[8]) {
  /* The frozen 50-digit-mpmath Q31 stage-8 words. */
  static const struct Complex twiddleValues[4] = {
      {INT32_MAX, 0},
      {1518500250, -1518500250},
      {0, INT32_MIN},
      {-1518500250, -1518500250},
  };
  int64_t even[4];
  int64_t odd[4];
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
      {"real-impulse", {{INT32_MAX, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}},
      /* Odd components land floor/saturation activity on every stage. */
      {"mixed",
       {{1200000001, -700000003},
        {-1600000000, 2100000000},
        {INT32_MIN, 500000005},
        {900000000, INT32_MAX},
        {-400000007, 1700000000},
        {2000000000, -2100000009},
        {-2100000000, -600000000},
        {1400000000, 800000011}}},
      {"extrema",
       {{INT32_MIN, INT32_MIN},
        {INT32_MAX, INT32_MAX},
        {INT32_MIN, INT32_MAX},
        {INT32_MAX, INT32_MIN},
        {INT32_MIN, 0},
        {0, INT32_MIN},
        {INT32_MAX, 0},
        {0, INT32_MAX}}},
  };

  for (unsigned caseIndex = 0; caseIndex < sizeof(cases) / sizeof(cases[0]); ++caseIndex) {
    int64_t input[8];
    int64_t expected[8];
    for (unsigned i = 0; i < 8; ++i)
      input[i] = pack(cases[caseIndex].input[i]);
    cfft8(input, expected);

    MemRefI64 inputRef = {input, input, 0, {8}, {1}};
    MemRefI64 output;
    _mlir_ciface_q31_cfft8(&output, &inputRef);
    if (output.sizes[0] != 8) {
      fprintf(stderr, "%s: expected output extent 8, got %" PRId64 "\n", cases[caseIndex].name,
              output.sizes[0]);
      free(output.allocated);
      return 1;
    }
    for (unsigned i = 0; i < 8; ++i) {
      int64_t actual = output.aligned[output.offset + i * output.strides[0]];
      if (actual != expected[i]) {
        fprintf(stderr, "%s[%u]: expected %016" PRIx64 ", got %016" PRIx64 "\n",
                cases[caseIndex].name, i, (uint64_t)expected[i], (uint64_t)actual);
        free(output.allocated);
        return 1;
      }
    }
    free(output.allocated);
  }
  return 0;
}

/* Independent type-II DCT reference over both fixed profiles.
 *
 * Independence is the point. The coefficients are quantized here from a
 * `long double` cosine WITHOUT the compiler's exact range reduction, so the
 * two derivations share no code and no rounding trick; at 64 mantissa bits the
 * unreduced angle still errs by about 5e-08 Q31 LSB, four orders below the
 * 4.9e-03 LSB worst tie margin, so agreement is evidence rather than
 * construction. Both shifts are derived from the extent and reached through
 * explicit floor division rather than the shift the lowering emits.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {
  int32_t *allocated;
  int32_t *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefI32;

typedef struct {
  int16_t *allocated;
  int16_t *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefI16;

extern void _mlir_ciface_dct8_q31(MemRefI32 *, MemRefI32 *);
extern void _mlir_ciface_dct8_q31_floor(MemRefI32 *, MemRefI32 *);
extern void _mlir_ciface_dct64_q31(MemRefI32 *, MemRefI32 *);
extern void _mlir_ciface_dct8_q15(MemRefI16 *, MemRefI16 *);

enum { kEven = 0, kFloor = 1 };

/* 2*(W-1) + log2(N) <= 62. The extents are powers of two, so floor and ceil
 * agree here and this arm cannot discriminate them; the product boundary is
 * discriminated by its declared rounding instead. */
static unsigned productShift(unsigned storageWidth, int64_t extent) {
  unsigned bits = 0;
  while (((int64_t)1 << (bits + 1)) <= extent)
    ++bits;
  unsigned exact = 2u * (storageWidth - 1u) + bits;
  return exact <= 62u ? 0u : exact - 62u;
}

/* Accumulator frac 2*(W-1)-p down to the declared reading W-2-m. */
static unsigned exportShift(unsigned storageWidth, int64_t extent) {
  unsigned bits = 0;
  while (((int64_t)1 << (bits + 1)) <= extent)
    ++bits;
  return storageWidth + bits - productShift(storageWidth, extent);
}

static int64_t roundShift(int64_t value, unsigned shift, int mode) {
  if (shift == 0)
    return value;
  int64_t divisor = (int64_t)1 << shift;
  int64_t quotient = value / divisor;
  int64_t remainder = value % divisor;
  if (remainder < 0) {
    --quotient;
    remainder += divisor;
  }
  if (mode == kFloor)
    return quotient;
  int64_t half = divisor >> 1;
  if (remainder > half)
    return quotient + 1;
  if (remainder < half)
    return quotient;
  return (quotient & 1) ? quotient + 1 : quotient;
}

static int64_t saturate(int64_t value, unsigned storageWidth) {
  int64_t top = ((int64_t)1 << (storageWidth - 1)) - 1;
  int64_t bottom = -((int64_t)1 << (storageWidth - 1));
  return value > top ? top : (value < bottom ? bottom : value);
}

static int64_t coefficient(int64_t extent, int64_t k, int64_t n, unsigned storageWidth) {
  static const long double kPi = 3.14159265358979323846264338327950288419716939937510582097494459L;
  long double angle = kPi * (long double)((2 * n + 1) * k) / (2.0L * (long double)extent);
  long double scaled = cosl(angle) * powl(2.0L, (long double)storageWidth - 1.0L);
  long double lower = floorl(scaled);
  long double fraction = scaled - lower;
  int64_t quantized = (int64_t)lower;
  if (fraction > 0.5L)
    ++quantized;
  else if (fraction == 0.5L && ((quantized & 1) != 0))
    ++quantized;
  return saturate(quantized, storageWidth);
}

static int64_t reference(const int64_t *input, int64_t extent, int64_t k, unsigned storageWidth,
                         int productMode, int exportMode) {
  unsigned narrow = productShift(storageWidth, extent);
  int64_t sum = 0;
  for (int64_t n = 0; n < extent; ++n) {
    int64_t product = input[n] * coefficient(extent, k, n, storageWidth);
    sum += roundShift(product, narrow, productMode);
  }
  return saturate(roundShift(sum, exportShift(storageWidth, extent), exportMode), storageWidth);
}

static uint32_t state = 0x2f6e2b1u;
static uint32_t nextRandom(void) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

static int failures = 0;

static void checkQ31(const char *name, void (*entry)(MemRefI32 *, MemRefI32 *), int64_t extent,
                     const int32_t *values, int productMode, int exportMode) {
  MemRefI32 in = {(int32_t *)values, (int32_t *)values, 0, {extent}, {1}};
  int32_t results[64];
  MemRefI32 out = {results, results, 0, {extent}, {1}};
  entry(&in, &out);
  int64_t wide[64];
  for (int64_t n = 0; n < extent; ++n)
    wide[n] = values[n];
  for (int64_t k = 0; k < extent; ++k) {
    int64_t expected = reference(wide, extent, k, 32, productMode, exportMode);
    int32_t got = results[k];
    if ((int64_t)got != expected) {
      printf("%s[%lld]: got %d expected %lld\n", name, (long long)k, got, (long long)expected);
      ++failures;
    }
  }
}

int main(void) {
  int32_t random8[8], random64[64];
  for (int i = 0; i < 8; ++i)
    random8[i] = (int32_t)nextRandom();
  for (int i = 0; i < 64; ++i)
    random64[i] = (int32_t)nextRandom();

  checkQ31("dct8_q31", _mlir_ciface_dct8_q31, 8, random8, kEven, kEven);
  checkQ31("dct8_q31_floor", _mlir_ciface_dct8_q31_floor, 8, random8, kFloor, kEven);
  checkQ31("dct64_q31", _mlir_ciface_dct64_q31, 64, random64, kEven, kEven);

  /* The two Q31 arms differ only in the declared product rounding, and at the
   * OUTPUT they agree here -- that is the contract, not a weak test. The two
   * accumulators differ by at most one unit per product, so by at most N, and
   * the export then divides by 2^32; a declared product mode reaches the
   * output only when the accumulator sits within N of an export tie, about
   * 2^-29 per element. What discriminates the narrowing itself is the rail
   * witness below, whose reference is unreachable without it. */
  {
    MemRefI32 in = {random8, random8, 0, {8}, {1}};
    int32_t evenValues[8], floorValues[8];
    MemRefI32 even = {evenValues, evenValues, 0, {8}, {1}};
    MemRefI32 floorArm = {floorValues, floorValues, 0, {8}, {1}};
    _mlir_ciface_dct8_q31(&in, &even);
    _mlir_ciface_dct8_q31_floor(&in, &floorArm);
    for (int k = 0; k < 8; ++k)
      if (evenValues[k] != floorValues[k])
        printf("note: product rounding reached the output at k=%d\n", k);
  }

  /* Named witness, and it is the ONLY arm that discriminates the derivation.
   * Measured by mutation: pinning the product shift to its N = 8 value leaves
   * dct8_q31 and the RANDOM dct64_q31 corpus passing, because the export shift
   * W + m - p compensates a wrong p exactly. The one consequence a wrong p
   * cannot hide is i64 overflow, and only this input reaches it: at N = 64 the
   * all-maximum input makes the k = 0 row sum 64 products of 2^62, which is
   * 2^68. Removing this arm would leave the extent dependence untested. */
  {
    int32_t rail[64];
    for (int i = 0; i < 64; ++i)
      rail[i] = INT32_MAX;
    checkQ31("dct64_q31_rail", _mlir_ciface_dct64_q31, 64, rail, kEven, kEven);
  }

  /* Q15 regression: the profile that carries no product boundary at all. */
  {
    int16_t values[8];
    for (int i = 0; i < 8; ++i)
      values[i] = (int16_t)nextRandom();
    MemRefI16 in = {values, values, 0, {8}, {1}};
    int16_t results[8];
    MemRefI16 out = {results, results, 0, {8}, {1}};
    _mlir_ciface_dct8_q15(&in, &out);
    int64_t wide[8];
    for (int n = 0; n < 8; ++n)
      wide[n] = values[n];
    for (int64_t k = 0; k < 8; ++k) {
      int64_t expected = reference(wide, 8, k, 16, kEven, kEven);
      if ((int64_t)results[k] != expected) {
        printf("dct8_q15[%lld]: got %d expected %lld\n", (long long)k, results[k],
               (long long)expected);
        ++failures;
      }
    }
  }

  if (failures != 0) {
    printf("%d mismatches\n", failures);
    return 1;
  }
  printf("dct fixed profiles agree\n");
  return 0;
}

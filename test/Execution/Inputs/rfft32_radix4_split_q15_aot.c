/* Independent reference and execution gate for the half-size radix-4 split
 * Q15 RFFT32 contract.
 *
 * The reference is written directly from the frozen contract equations with
 * explicit floor division and explicit clamps, independently of the compiler
 * lowering. One fixed vector additionally carries golden packed bins and is
 * known to activate the schedule's stage-two saturation, so both the exact
 * arithmetic and the only reachable clamp are anchored here.
 */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

extern int32_t rfft32_radix4_split_q15_value(int16_t, int16_t, int16_t, int16_t, int16_t, int16_t,
                                             int16_t, int16_t, int16_t, int16_t, int16_t, int16_t,
                                             int16_t, int16_t, int16_t, int16_t, int16_t, int16_t,
                                             int16_t, int16_t, int16_t, int16_t, int16_t, int16_t,
                                             int16_t, int16_t, int16_t, int16_t, int16_t, int16_t,
                                             int16_t, int16_t, intptr_t);

/* floor(value / 2^shift) without relying on implementation-defined right
 * shifts of negative values. */
static int64_t floorShift(int64_t value, int shift) {
  int64_t divisor = (int64_t)1 << shift;
  int64_t quotient = value / divisor;
  if (value % divisor != 0 && value < 0)
    --quotient;
  return quotient;
}

static int64_t sat16(int64_t value) {
  if (value > 32767)
    return 32767;
  if (value < -32768)
    return -32768;
  return value;
}

/* Frozen Q15 twiddle pairs (pair index -> co, si); pairs 5, 7, and 8 are
 * never consumed. */
static const int32_t kTwiddles[10][2] = {
    {32767, 0}, {30273, 12539},  {23170, 23170}, {12539, 30273}, {0, 32767},
    {0, 0},     {-23171, 23170}, {0, 0},         {0, 0},         {-30274, -12540}};

/* Frozen split coefficients (bin -> Ar, Ai, Br, Bi); bin 0 is unused. */
static const int32_t kSplit[16][4] = {{0, 0, 0, 0},
                                      {13188, -16069, 19580, 16069},
                                      {10114, -15137, 22654, 15137},
                                      {7282, -13623, 25486, 13623},
                                      {4799, -11585, 27969, 11585},
                                      {2761, -9102, 30007, 9102},
                                      {1247, -6270, 31521, 6270},
                                      {315, -3196, 32453, 3196},
                                      {0, 0, 32767, 0},
                                      {315, 3196, 32453, -3196},
                                      {1247, 6270, 31521, -6270},
                                      {2761, 9102, 30007, -9102},
                                      {4799, 11585, 27969, -11585},
                                      {7282, 13623, 25486, -13623},
                                      {10114, 15137, 22654, -15137},
                                      {13188, 16069, 19580, -16069}};

static const int kBitReversalPairs[6][2] = {{1, 8}, {2, 4}, {3, 12}, {5, 10}, {7, 14}, {11, 13}};

/* Reinterpret an unsigned bit pattern as its two's-complement value through
 * explicit arithmetic instead of an implementation-defined narrowing. */
static int16_t toSigned16(uint32_t bits) {
  uint32_t low = bits & 0xFFFFu;
  return low <= 32767u ? (int16_t)low : (int16_t)((int32_t)low - 65536);
}

static int32_t toSigned32(uint32_t bits) {
  return bits <= (uint32_t)INT32_MAX ? (int32_t)bits
                                     : (int32_t)((int64_t)bits - (INT64_C(1) << 32));
}

static int32_t packBin(int64_t real, int64_t imaginary) {
  uint32_t realBits = (uint32_t)((uint64_t)real & 0xFFFFu);
  uint32_t imaginaryBits = (uint32_t)((uint64_t)imaginary & 0xFFFFu);
  return toSigned32((imaginaryBits << 16) | realBits);
}

static void referenceRfft32(const int16_t *x, int32_t *bins) {
  int64_t re[16], im[16];
  for (int m = 0; m < 16; ++m) {
    re[m] = x[2 * m];
    im[m] = x[2 * m + 1];
  }

  /* Stage 1: radix-4 groups (g, g+4, g+8, g+12), twiddle pair index g. */
  for (int g = 0; g < 4; ++g) {
    int a = g, b = g + 4, c = g + 8, d = g + 12;
    int64_t t0 = floorShift(re[a], 2), t1 = floorShift(im[a], 2);
    int64_t s0 = floorShift(re[c], 2), s1 = floorShift(im[c], 2);
    int64_t sum0 = t0 + s0, sum1 = t1 + s1;
    int64_t diff0 = t0 - s0, diff1 = t1 - s1;
    int64_t tb0 = floorShift(re[b], 2), tb1 = floorShift(im[b], 2);
    int64_t u0 = floorShift(re[d], 2), u1 = floorShift(im[d], 2);
    int64_t tSum0 = tb0 + u0, tSum1 = tb1 + u1;
    re[a] = floorShift(sum0, 1) + floorShift(tSum0, 1);
    im[a] = floorShift(sum1, 1) + floorShift(tSum1, 1);
    int64_t r0 = sum0 - tSum0, r1 = sum1 - tSum1;
    const int32_t *w2 = kTwiddles[2 * g];
    re[b] = floorShift(w2[0] * r0 + w2[1] * r1, 16);
    im[b] = floorShift(w2[0] * r1 - w2[1] * r0, 16);
    int64_t tDiff0 = tb0 - u0, tDiff1 = tb1 - u1;
    int64_t rr0 = diff0 - tDiff1, rr1 = diff1 + tDiff0;
    int64_t ss0 = diff0 + tDiff1, ss1 = diff1 - tDiff0;
    const int32_t *w1 = kTwiddles[g];
    re[c] = floorShift(w1[0] * ss0 + w1[1] * ss1, 16);
    im[c] = floorShift(w1[0] * ss1 - w1[1] * ss0, 16);
    const int32_t *w3 = kTwiddles[3 * g];
    re[d] = floorShift(w3[0] * rr0 + w3[1] * rr1, 16);
    im[d] = floorShift(w3[0] * rr1 - w3[1] * rr0, 16);
  }

  /* Stage 2: unit-twiddle groups; the only reachable saturation points. */
  for (int g = 0; g < 16; g += 4) {
    int64_t ar = re[g], ai = im[g], br = re[g + 1], bi = im[g + 1];
    int64_t cr = re[g + 2], ci = im[g + 2], dr = re[g + 3], di = im[g + 3];
    int64_t r0 = sat16(ar + cr), r1 = sat16(ai + ci);
    int64_t s0 = sat16(ar - cr), s1 = sat16(ai - ci);
    int64_t t0 = sat16(br + dr), t1 = sat16(bi + di);
    re[g] = floorShift(r0, 1) + floorShift(t0, 1);
    im[g] = floorShift(r1, 1) + floorShift(t1, 1);
    re[g + 1] = floorShift(r0, 1) - floorShift(t0, 1);
    im[g + 1] = floorShift(r1, 1) - floorShift(t1, 1);
    int64_t d0 = sat16(br - dr), d1 = sat16(bi - di);
    re[g + 2] = floorShift(s0, 1) + floorShift(d1, 1);
    im[g + 2] = floorShift(s1, 1) - floorShift(d0, 1);
    re[g + 3] = floorShift(s0, 1) - floorShift(d1, 1);
    im[g + 3] = floorShift(s1, 1) + floorShift(d0, 1);
  }

  for (int p = 0; p < 6; ++p) {
    int a = kBitReversalPairs[p][0], b = kBitReversalPairs[p][1];
    int64_t swap = re[a];
    re[a] = re[b];
    re[b] = swap;
    swap = im[a];
    im[a] = im[b];
    im[b] = swap;
  }

  /* Split stage into compact natural-order bins 0..16. */
  for (int k = 1; k < 16; ++k) {
    const int32_t *cf = kSplit[k];
    int64_t accR = re[k] * cf[0] - im[k] * cf[1] + re[16 - k] * cf[2] + im[16 - k] * cf[3];
    int64_t accI = re[16 - k] * cf[3] - im[16 - k] * cf[2] + im[k] * cf[0] + re[k] * cf[1];
    bins[k] = packBin(floorShift(accR, 16), floorShift(accI, 16));
  }
  bins[0] = packBin(floorShift(re[0] + im[0], 1), 0);
  bins[16] = packBin(floorShift(re[0] - im[0], 1), 0);
}

static int failures = 0;

static void checkVector(const char *name, const int16_t *x, const int32_t *golden) {
  int32_t expected[17];
  referenceRfft32(x, expected);
  for (int k = 0; k < 17; ++k) {
    if (golden && expected[k] != golden[k]) {
      printf("FAIL %s: reference bin %d = %" PRId32 " but golden = %" PRId32 "\n", name, k,
             expected[k], golden[k]);
      ++failures;
    }
    int32_t actual = rfft32_radix4_split_q15_value(
        x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13],
        x[14], x[15], x[16], x[17], x[18], x[19], x[20], x[21], x[22], x[23], x[24], x[25], x[26],
        x[27], x[28], x[29], x[30], x[31], (intptr_t)k);
    if (actual != expected[k]) {
      printf("FAIL %s: bin %d = %" PRId32 " but reference = %" PRId32 "\n", name, k, actual,
             expected[k]);
      ++failures;
    }
  }
}

static uint32_t nextRandom(uint32_t *state) {
  uint32_t value = *state;
  value ^= value << 13;
  value ^= value >> 17;
  value ^= value << 5;
  *state = value;
  return value;
}

int main(void) {
  int16_t x[32];
  char name[64];

  const int16_t zeros[32] = {0};
  checkVector("zeros", zeros, NULL);

  for (int n = 0; n < 32; ++n)
    x[n] = 32767;
  checkVector("positive full scale", x, NULL);

  for (int n = 0; n < 32; ++n)
    x[n] = -32768;
  checkVector("negative full scale", x, NULL);

  for (int n = 0; n < 32; ++n)
    x[n] = (n & 1) ? -32768 : 32767;
  checkVector("alternating full scale", x, NULL);

  /* Fixed golden vector; activates one stage-two saturating clamp. */
  static const int16_t witness[32] = {
      -32768, 32767, 32767, -32768, 32767, -32768, 32767,  -32768, -32768, 32767,  32767,
      -32768, 32767, 32767, -32768, 32767, 32767,  -32768, -32768, 32767,  -32768, 32767,
      -32768, 32767, 32767, 32767,  32767, 32767,  32767,  -32768, -32768, 32767};
  static const int32_t witnessBins[17] = {
      6140,      192545046,  236975279, -422773805, -268434608, -145226781,
      113773392, -223345280, 2047,      205971688,  382208848,  -441849483,
      268430511, 164421682,  -31460176, 464719648,  63488};
  checkVector("stage-two saturation witness", witness, witnessBins);

  for (int position = 0; position < 32; ++position) {
    for (int n = 0; n < 32; ++n)
      x[n] = 0;
    x[position] = 32767;
    snprintf(name, sizeof(name), "positive impulse %d", position);
    checkVector(name, x, NULL);
    x[position] = -32768;
    snprintf(name, sizeof(name), "negative impulse %d", position);
    checkVector(name, x, NULL);
  }

  uint32_t state = 0x20260728u;
  for (int trial = 0; trial < 256; ++trial) {
    for (int n = 0; n < 32; ++n)
      x[n] = toSigned16(nextRandom(&state));
    snprintf(name, sizeof(name), "random %d", trial);
    checkVector(name, x, NULL);
  }
  for (int trial = 0; trial < 64; ++trial) {
    for (int n = 0; n < 32; ++n)
      x[n] = (nextRandom(&state) & 1u) ? 32767 : -32768;
    snprintf(name, sizeof(name), "random full scale %d", trial);
    checkVector(name, x, NULL);
  }

  if (failures != 0) {
    printf("rfft32 radix-4 split execution gate failed: %d\n", failures);
    return 1;
  }
  printf("rfft32 radix-4 split execution gate passed\n");
  return 0;
}

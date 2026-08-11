#include <assert.h>
#include <stdint.h>
#include <stdio.h>

extern int32_t mul_ntp_wrap_s1(int32_t v, int32_t w);
extern int32_t mul_floor_wrap_s1(int32_t v, int32_t w);
extern int32_t mul_ntp_wrap_s0(int32_t v, int32_t w);
extern int32_t mul_ntp_sat_s18(int32_t v, int32_t w);
extern int32_t mul_floor_wrap_s15(int32_t v, int32_t w);
extern int32_t mul_floor_sat_s15(int32_t v, int32_t w);
extern int32_t mul_swapped_floor_wrap_s0(int32_t v, int32_t w);
extern int64_t bfly_plain_floor_wrap_s1(int32_t a, int32_t b);
extern int64_t bfly_plain_ntp_wrap_s1(int32_t a, int32_t b);
extern int64_t bfly_plain_ntp_sat_s1(int32_t a, int32_t b);
extern int64_t bfly_plain_floor_sat_s0(int32_t a, int32_t b);
extern int64_t bfly_cross_floor_wrap_s1(int32_t a, int32_t b);
extern int64_t bfly_cross_ntp_sat_s0(int32_t a, int32_t b);
extern int64_t compose_pm_j(int32_t x, int32_t c, int32_t w);

static int64_t lo16(uint32_t v) { return (int16_t)(v & 0xFFFF); }
static int64_t hi16(uint32_t v) { return (int16_t)((v >> 16) & 0xFFFF); }

static uint32_t ref_mode16(int64_t value, unsigned k, int rnd, int sat) {
  int64_t shifted;
  if (k == 0)
    shifted = value;
  else if (rnd)
    shifted = (value + ((int64_t)1 << (k - 1))) >> k;
  else
    shifted = value >> k;
  if (sat) {
    if (shifted > 32767)
      shifted = 32767;
    if (shifted < -32768)
      shifted = -32768;
  }
  return (uint32_t)shifted & 0xFFFF;
}

static uint32_t ref_mul_conj(uint32_t v, uint32_t w, unsigned k, int rnd, int sat, int swap) {
  int64_t real = lo16(v) * lo16(w) + hi16(v) * hi16(w);
  int64_t imag = hi16(v) * lo16(w) - lo16(v) * hi16(w);
  uint32_t hi = swap ? ref_mode16(real, k, rnd, sat) : ref_mode16(imag, k, rnd, sat);
  uint32_t lo = swap ? ref_mode16(imag, k, rnd, sat) : ref_mode16(real, k, rnd, sat);
  return (hi << 16) | lo;
}

static uint64_t ref_bfly(uint32_t a, uint32_t b, unsigned k, int rnd, int sat, int cross) {
  int64_t sum_r = lo16(a) + lo16(b), diff_r = lo16(a) - lo16(b);
  int64_t sum_i = hi16(a) + hi16(b), diff_i = hi16(a) - hi16(b);
  uint32_t out0 =
      (ref_mode16(cross ? diff_i : sum_i, k, rnd, sat) << 16) | ref_mode16(sum_r, k, rnd, sat);
  uint32_t out1 =
      (ref_mode16(cross ? sum_i : diff_i, k, rnd, sat) << 16) | ref_mode16(diff_r, k, rnd, sat);
  return ((uint64_t)out1 << 32) | out0;
}

static int failures = 0;
static int checks = 0;

static void expect32(const char *name, uint32_t got, uint32_t want) {
  ++checks;
  if (got != want) {
    ++failures;
    fprintf(stderr, "FAIL %s: got 0x%08x want 0x%08x\n", name, got, want);
  }
}

static void expect64(const char *name, uint64_t got, uint64_t want) {
  ++checks;
  if (got != want) {
    ++failures;
    fprintf(stderr, "FAIL %s: got 0x%016llx want 0x%016llx\n", name, (unsigned long long)got,
            (unsigned long long)want);
  }
}

int main(void) {
  /* Boundary witnesses pin the reference before the differential runs. */
  assert((ref_mul_conj(0x0000FFFF, 0x00000003, 1, 1, 0, 0) & 0xFFFF) == 0xFFFF);
  assert((ref_mul_conj(0x0000FFFF, 0x00000003, 1, 0, 0, 0) & 0xFFFF) == 0xFFFE);
  assert((ref_mul_conj(0x0000FFFF, 0x00000003, 0, 1, 0, 0) & 0xFFFF) == 0xFFFD);
  assert((ref_mul_conj(0x7FFF7FFF, 0x7FFF7FFF, 18, 1, 1, 0) & 0xFFFF) == 0x2000);

  static const uint32_t edges[] = {
      0x80008000, 0x7FFF7FFF, 0x80007FFF, 0x7FFF8000, 0x0000FFFF, 0x00000003,
      0xFFFF0000, 0x00010001, 0x40004000, 0xC000C000, 0x00008000, 0x7FFF0001,
  };
  const int edgeCount = (int)(sizeof(edges) / sizeof(edges[0]));

  uint32_t state = 0xC0FFEE01u;
  uint32_t corpus[288 + 64];
  int corpusCount = 0;
  for (int i = 0; i < edgeCount; ++i)
    for (int j = 0; j < edgeCount; ++j) {
      corpus[corpusCount++] = edges[i];
      corpus[corpusCount++] = edges[j];
    }
  for (int i = 0; i < 32; ++i) {
    state = state * 1664525u + 1013904223u;
    corpus[corpusCount++] = state;
    state = state * 1664525u + 1013904223u;
    corpus[corpusCount++] = state;
  }

  for (int i = 0; i + 1 < corpusCount; i += 2) {
    uint32_t v = corpus[i], w = corpus[i + 1];
    expect32("mul_ntp_wrap_s1", (uint32_t)mul_ntp_wrap_s1((int32_t)v, (int32_t)w),
             ref_mul_conj(v, w, 1, 1, 0, 0));
    expect32("mul_floor_wrap_s1", (uint32_t)mul_floor_wrap_s1((int32_t)v, (int32_t)w),
             ref_mul_conj(v, w, 1, 0, 0, 0));
    expect32("mul_ntp_wrap_s0", (uint32_t)mul_ntp_wrap_s0((int32_t)v, (int32_t)w),
             ref_mul_conj(v, w, 0, 1, 0, 0));
    expect32("mul_ntp_sat_s18", (uint32_t)mul_ntp_sat_s18((int32_t)v, (int32_t)w),
             ref_mul_conj(v, w, 18, 1, 1, 0));
    expect32("mul_floor_wrap_s15", (uint32_t)mul_floor_wrap_s15((int32_t)v, (int32_t)w),
             ref_mul_conj(v, w, 15, 0, 0, 0));
    expect32("mul_floor_sat_s15", (uint32_t)mul_floor_sat_s15((int32_t)v, (int32_t)w),
             ref_mul_conj(v, w, 15, 0, 1, 0));
    expect32("mul_swapped_floor_wrap_s0",
             (uint32_t)mul_swapped_floor_wrap_s0((int32_t)v, (int32_t)w),
             ref_mul_conj(v, w, 0, 0, 0, 1));
    expect64("bfly_plain_floor_wrap_s1", (uint64_t)bfly_plain_floor_wrap_s1((int32_t)v, (int32_t)w),
             ref_bfly(v, w, 1, 0, 0, 0));
    expect64("bfly_plain_ntp_wrap_s1", (uint64_t)bfly_plain_ntp_wrap_s1((int32_t)v, (int32_t)w),
             ref_bfly(v, w, 1, 1, 0, 0));
    expect64("bfly_plain_ntp_sat_s1", (uint64_t)bfly_plain_ntp_sat_s1((int32_t)v, (int32_t)w),
             ref_bfly(v, w, 1, 1, 1, 0));
    expect64("bfly_plain_floor_sat_s0", (uint64_t)bfly_plain_floor_sat_s0((int32_t)v, (int32_t)w),
             ref_bfly(v, w, 0, 0, 1, 0));
    expect64("bfly_cross_floor_wrap_s1", (uint64_t)bfly_cross_floor_wrap_s1((int32_t)v, (int32_t)w),
             ref_bfly(v, w, 1, 0, 0, 1));
    expect64("bfly_cross_ntp_sat_s0", (uint64_t)bfly_cross_ntp_sat_s0((int32_t)v, (int32_t)w),
             ref_bfly(v, w, 0, 1, 1, 1));
  }

  /* The -+j composition against exact complex arithmetic: small operands,
     shift 0, no wrap, so X -+ j*P is bit-exact end to end. */
  static const int16_t smalls[][6] = {
      {0x10, 0x20, 2, 3, 4, 5},
      {-7, 9, -3, 6, 8, -2},
      {100, -50, 12, -34, 56, 78},
  };
  for (int i = 0; i < 3; ++i) {
    int16_t xr = smalls[i][0], xi = smalls[i][1], cr = smalls[i][2];
    int16_t ci = smalls[i][3], wr = smalls[i][4], wi = smalls[i][5];
    int32_t x = (int32_t)(((uint32_t)(uint16_t)xi << 16) | (uint16_t)xr);
    int32_t c = (int32_t)(((uint32_t)(uint16_t)ci << 16) | (uint16_t)cr);
    int32_t w = (int32_t)(((uint32_t)(uint16_t)wi << 16) | (uint16_t)wr);
    int32_t pr = cr * wr + ci * wi, pi = ci * wr - cr * wi;
    uint32_t minus = (((uint32_t)(uint16_t)(xi - pr)) << 16) | (uint16_t)(xr + pi);
    uint32_t plus = (((uint32_t)(uint16_t)(xi + pr)) << 16) | (uint16_t)(xr - pi);
    expect64("compose_pm_j", (uint64_t)compose_pm_j(x, c, w), ((uint64_t)plus << 32) | minus);
  }

  if (failures != 0) {
    fprintf(stderr, "ortumcore cx packed differential: %d/%d FAILED\n", failures, checks);
    return 1;
  }
  printf("ortumcore cx packed differential: PASS (%d checks incl. witness card, "
         "swap layout, cross variant, and the -+j composition)\n",
         checks);
  return 0;
}

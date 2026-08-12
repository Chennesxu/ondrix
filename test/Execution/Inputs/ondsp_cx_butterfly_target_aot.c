#include <stdint.h>
#include <stdio.h>

extern int64_t bf_floor_wrap(int32_t a, int32_t b);
extern int64_t bf_ntp_sat(int32_t a, int32_t b);
extern int64_t bf_ntp_wrap_out0(int32_t a, int32_t b);

#define TW_FLOOR_WRAP 0x5A82A57Eu
#define TW_NTP_SAT 0x00030003u
#define TW_NTP_WRAP 0x7FFF8000u

static int64_t lo16(uint32_t v) { return (int16_t)(v & 0xFFFF); }
static int64_t hi16(uint32_t v) { return (int16_t)((v >> 16) & 0xFFFF); }

static uint32_t ref_scale(int64_t value, unsigned k, int rnd, int sat) {
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

/* The ondsp contract: t = requant(b*w), outputs requant(a +- t). */
static uint64_t ref_butterfly(uint32_t a, uint32_t b, uint32_t w, unsigned outShift, int rnd,
                              int sat) {
  int64_t pr = lo16(b) * lo16(w) - hi16(b) * hi16(w);
  int64_t pi = lo16(b) * hi16(w) + hi16(b) * lo16(w);
  int64_t tr = (int16_t)ref_scale(pr, 15, rnd, sat);
  int64_t ti = (int16_t)ref_scale(pi, 15, rnd, sat);
  uint32_t out0 = (ref_scale(hi16(a) + ti, outShift, rnd, sat) << 16) |
                  ref_scale(lo16(a) + tr, outShift, rnd, sat);
  uint32_t out1 = (ref_scale(hi16(a) - ti, outShift, rnd, sat) << 16) |
                  ref_scale(lo16(a) - tr, outShift, rnd, sat);
  return ((uint64_t)out1 << 32) | out0;
}

static int failures = 0;
static int checks = 0;

static void expect(const char *name, uint64_t got, uint64_t want) {
  ++checks;
  if (got != want) {
    ++failures;
    fprintf(stderr, "FAIL %s: got 0x%016llx want 0x%016llx\n", name, (unsigned long long)got,
            (unsigned long long)want);
  }
}

int main(void) {
  static const uint32_t edges[] = {
      0x80008000, 0x7FFF7FFF, 0x80007FFF, 0x7FFF8000, 0x0000FFFF, 0x00010001,
      0x40004000, 0xC000C000, 0x00008000, 0x7FFF0001, 0x00004000, 0xFFFF4000,
  };
  const int edgeCount = (int)(sizeof(edges) / sizeof(edges[0]));

  uint32_t state = 0xB0BACAFEu;
  for (int i = 0; i < edgeCount * edgeCount + 48; ++i) {
    uint32_t a, b;
    if (i < edgeCount * edgeCount) {
      a = edges[i / edgeCount];
      b = edges[i % edgeCount];
    } else {
      state = state * 1664525u + 1013904223u;
      a = state;
      state = state * 1664525u + 1013904223u;
      b = state;
    }
    expect("bf_floor_wrap", (uint64_t)bf_floor_wrap((int32_t)a, (int32_t)b),
           ref_butterfly(a, b, TW_FLOOR_WRAP, 1, 0, 0));
    expect("bf_ntp_sat", (uint64_t)bf_ntp_sat((int32_t)a, (int32_t)b),
           ref_butterfly(a, b, TW_NTP_SAT, 1, 1, 1));
    expect("bf_ntp_wrap_out0", (uint64_t)bf_ntp_wrap_out0((int32_t)a, (int32_t)b),
           ref_butterfly(a, b, TW_NTP_WRAP, 0, 1, 0));
  }

  if (failures != 0) {
    fprintf(stderr, "cx_butterfly target differential: %d/%d FAILED\n", failures, checks);
    return 1;
  }
  printf("cx_butterfly target differential: PASS (%d checks; floor/NTP, wrap/sat, "
         "output shift 0 and 1, rail twiddle)\n",
         checks);
  return 0;
}

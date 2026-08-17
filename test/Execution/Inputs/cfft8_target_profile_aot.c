#include <stdint.h>
#include <stdio.h>

extern int32_t cfft8_floor_wrap(int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t,
                                int32_t, int64_t);
extern int32_t cfft8_ntp_sat(int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t,
                             int64_t);
extern int32_t cfft8_floor_sat(int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t,
                               int32_t, int64_t);
extern int32_t cfft8_ntp_wrap(int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t,
                              int32_t, int64_t);

static int64_t lo16(uint32_t v) { return (int16_t)(v & 0xFFFF); }
static int64_t hi16(uint32_t v) { return (int16_t)((v >> 16) & 0xFFFF); }

static uint32_t scale16(int64_t value, unsigned k, int rnd, int sat) {
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

/* The contract's quantized forward twiddles: W(2)=[1]; W(4)=[1,-j];
   W(8)=[1, 23170-j23170, -j, -23170-j23170]; +1 is the positive maximum. */
static const int16_t TW_RE[3][4] = {{32767, 0, 0, 0}, {32767, 0, 0, 0}, {32767, 23170, 0, -23170}};
static const int16_t TW_IM[3][4] = {{0, 0, 0, 0}, {0, -32768, 0, 0}, {0, -23170, -32768, -23170}};

static int levelOf(int size) { return size == 2 ? 0 : size == 4 ? 1 : 2; }

static void ref_cfft(int size, const uint32_t *in, uint32_t *out, int rnd, int sat) {
  if (size == 1) {
    out[0] = in[0];
    return;
  }
  uint32_t evens_in[4], odds_in[4], evens[4], odds[4];
  for (int i = 0; i < size / 2; ++i) {
    evens_in[i] = in[2 * i];
    odds_in[i] = in[2 * i + 1];
  }
  ref_cfft(size / 2, evens_in, evens, rnd, sat);
  ref_cfft(size / 2, odds_in, odds, rnd, sat);
  /* Inventory pairing (decision 2026-08-17): legs k and k + size/4 share
     the twiddle W(size, k); the second leg is the cross combine a -+ j*t
     over its own requantized product, so -j never multiplies. */
  int plainLegs = size == 2 ? 1 : size / 4;
  for (int k = 0; k < plainLegs; ++k) {
    int64_t wr = TW_RE[levelOf(size)][k], wi = TW_IM[levelOf(size)][k];
    int64_t pr = lo16(odds[k]) * wr - hi16(odds[k]) * wi;
    int64_t pi = lo16(odds[k]) * wi + hi16(odds[k]) * wr;
    int64_t tr = (int16_t)scale16(pr, 15, rnd, sat);
    int64_t ti = (int16_t)scale16(pi, 15, rnd, sat);
    out[k] = (scale16(hi16(evens[k]) + ti, 1, rnd, sat) << 16) |
             scale16(lo16(evens[k]) + tr, 1, rnd, sat);
    out[k + size / 2] = (scale16(hi16(evens[k]) - ti, 1, rnd, sat) << 16) |
                        scale16(lo16(evens[k]) - tr, 1, rnd, sat);
    if (size == 2)
      continue;
    int kc = k + size / 4;
    int64_t cr = lo16(odds[kc]) * wr - hi16(odds[kc]) * wi;
    int64_t ci = lo16(odds[kc]) * wi + hi16(odds[kc]) * wr;
    int64_t ur = (int16_t)scale16(cr, 15, rnd, sat);
    int64_t ui = (int16_t)scale16(ci, 15, rnd, sat);
    out[kc] = (scale16(hi16(evens[kc]) - ur, 1, rnd, sat) << 16) |
              scale16(lo16(evens[kc]) + ui, 1, rnd, sat);
    out[kc + size / 2] = (scale16(hi16(evens[kc]) + ur, 1, rnd, sat) << 16) |
                         scale16(lo16(evens[kc]) - ui, 1, rnd, sat);
  }
}

static const uint32_t VECTORS[3][8] = {
    {0x10002000, 0x03000400, 0x7FFF7FFF, 0x80008000, 0x00010001, 0xFFFF4000, 0x5A82A57E,
     0x00030002},
    {0x7FFF0001, 0x8000FFFF, 0x40004000, 0xC000C000, 0x00008000, 0x00000003, 0x12345678,
     0x9ABCDEF0},
    {0x00640032, 0xFF9CFFCE, 0x30001000, 0xD000F000, 0x7FFF8000, 0x80007FFF, 0x00000000,
     0xFFFFFFFF},
};

static int failures = 0;
static int checks = 0;

int main(void) {
  struct Config {
    const char *name;
    int32_t (*fn)(int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int64_t);
    int rnd;
    int sat;
  } configs[] = {
      {"floor_wrap", cfft8_floor_wrap, 0, 0},
      {"ntp_sat", cfft8_ntp_sat, 1, 1},
      {"floor_sat", cfft8_floor_sat, 0, 1},
      {"ntp_wrap", cfft8_ntp_wrap, 1, 0},
  };
  for (int v = 0; v < 3; ++v) {
    const uint32_t *x = VECTORS[v];
    for (unsigned c = 0; c < sizeof(configs) / sizeof(configs[0]); ++c) {
      uint32_t want[8];
      ref_cfft(8, x, want, configs[c].rnd, configs[c].sat);
      for (int k = 0; k < 8; ++k) {
        uint32_t got =
            (uint32_t)configs[c].fn((int32_t)x[0], (int32_t)x[1], (int32_t)x[2], (int32_t)x[3],
                                    (int32_t)x[4], (int32_t)x[5], (int32_t)x[6], (int32_t)x[7], k);
        ++checks;
        if (got != want[k]) {
          ++failures;
          fprintf(stderr, "FAIL %s v%d[%d]: got 0x%08x want 0x%08x\n", configs[c].name, v, k, got,
                  want[k]);
        }
      }
    }
  }
  if (failures != 0) {
    fprintf(stderr, "cfft8 target profile differential: %d/%d FAILED\n", failures, checks);
    return 1;
  }
  printf("cfft8 target profile differential: PASS (%d checks; all four "
         "inventory whole-transform profiles)\n",
         checks);
  return 0;
}

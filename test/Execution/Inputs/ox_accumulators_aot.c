#include <stdint.h>
#include <stdio.h>

#define N 1200

static uint64_t state = 0x9e3779b97f4a7c15u;

static uint64_t next(void) {
  state ^= state << 13;
  state ^= state >> 7;
  state ^= state << 17;
  return state;
}

static int64_t saturate(int64_t value, int bits) {
  int64_t top = (INT64_C(1) << (bits - 1)) - 1;
  return value > top ? top : value < -top - 1 ? -top - 1 : value;
}

// The low `bits` bits of value, sign-extended.
static int64_t wrap(int64_t value, int bits) {
  uint64_t span = UINT64_C(1) << bits;
  uint64_t low = (uint64_t)value & (span - 1);
  return low >= span / 2 ? (int64_t)(low - span) : (int64_t)low;
}

// value / 2^15 rounded to nearest, ties toward +infinity or to even.
static int64_t round15(int64_t value, int even) {
  int64_t floor = (value - (value & 32767)) / 32768;
  int64_t rest = value & 32767;
  if (rest > 16384 || (rest == 16384 && (!even || (floor & 1))))
    ++floor;
  return floor;
}

static int16_t mult15(int16_t a, int16_t b) {
  return (int16_t)saturate(round15((int64_t)a * b, 0), 16);
}

static int fail(const char *what, unsigned n) {
  fprintf(stderr, "%s %u\n", what, n);
  return 1;
}

int main(void) {
  static int16_t x[N], h[N];
  static int32_t wx[N], wh[N];
  for (unsigned i = 0; i < N; ++i) {
    x[i] = (int16_t)(next() >> 48);
    h[i] = (int16_t)(next() >> 48);
    wx[i] = (int32_t)(next() >> 32);
    wh[i] = (int32_t)(next() >> 32);
  }
  // 0.5 is an export tie; +1 then -1 runs push a partial sum past 1 and back;
  // 660 products of exactly +1 leave the 40-bit range, and twice 2^62 the 64-bit one.
  x[0] = 1, h[0] = 16384;
  for (unsigned i = 20; i < 40; ++i)
    x[i] = INT16_MIN, h[i] = i < 30 ? INT16_MIN : INT16_MAX;
  for (unsigned i = 40; i < 700; ++i)
    x[i] = h[i] = INT16_MIN;
  for (unsigned i = 10; i < 12; ++i)
    wx[i] = wh[i] = INT32_MIN;

  int q31Differs = 0, saturationDiffers = 0, poleDiffers = 0;
  for (unsigned n = 0; n <= N; ++n) {
    int64_t sat = 0, wrapped = 0, narrow = 0;
    int16_t pole = 0, twice = 0;
    __int128 wide = 0;
    for (unsigned i = 0; i < n; ++i) {
      int64_t product = (int64_t)x[i] * h[i];
      sat = saturate(sat + product, 40);
      wrapped = wrap(wrapped + product, 40);
      narrow = saturate(narrow + saturate(product * 2, 32), 32);
      int64_t acc = (int64_t)pole * 29491 + (int64_t)x[i] * 3277;
      pole = (int16_t)saturate(round15(acc, 0), 16);
      twice = (int16_t)saturate((int64_t)mult15(twice, 29491) + mult15(x[i], 3277), 16);
      wide += (__int128)wx[i] * wh[i];
      if (wide > INT64_MAX)
        wide = INT64_MAX;
      if (wide < INT64_MIN)
        wide = INT64_MIN;
    }
    int16_t expected = (int16_t)saturate(round15(sat, 0), 16);
    int16_t expectedWrap = (int16_t)saturate(round15(wrapped, 1), 16);
    q31Differs |= saturate(round15(narrow / 2, 0), 16) != expected;
    saturationDiffers |= saturate(round15(wrapped, 0), 16) != expected;
    poleDiffers |= pole != twice;
    if (ondrix_acc_mac(x, n, h, n) != expected || ondrix_acc_dot(x, h, n) != expected)
      return fail("saturate", n);
    if (ondrix_acc_wrap(x, n, h, n) != expectedWrap || ondrix_acc_dot_wrap(x, h, n) != expectedWrap)
      return fail("wrap", n);
    if (ondrix_acc_pole(x, n) != pole)
      return fail("pole", n);
    if (ondrix_acc_q31(wx, n, wh, n) != (int32_t)wrap((int64_t)(wide / ((__int128)1 << 31)), 32))
      return fail("q31", n);
  }
  static const int16_t gains[] = {3277, 16384, INT16_MAX, INT16_MIN};
  for (unsigned g = 0; g < 4; ++g) {
    int16_t y = 0;
    for (unsigned n = 0; n <= N; ++n) {
      if (ondrix_acc_leak(x, n, gains[g]) != y)
        return fail("leak", n);
      if (n < N) {
        int64_t acc = (int64_t)y * 32768 + (int64_t)gains[g] * x[n] - (int64_t)gains[g] * y;
        y = (int16_t)saturate(round15(wrap(acc, 40), 1), 16);
      }
    }
  }
  if (!q31Differs || !saturationDiffers || !poleDiffers)
    return fail("witness", 0);
  return 0;
}

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uint64_t state = 0xa54ff53a5f1d36f1u;

static uint64_t next(void) {
  state ^= state << 13;
  state ^= state >> 7;
  state ^= state << 17;
  return state;
}

static int fail(const char *what, long n) {
  fprintf(stderr, "%s %ld\n", what, n);
  return 1;
}

static int same(const void *a, const void *b, size_t size) { return memcmp(a, b, size) == 0; }

static float real(void) { return (float)((int32_t)(next() >> 40) - (1 << 23)) / 65536.0f; }

static int16_t sat16(int32_t v) {
  return v > INT16_MAX ? INT16_MAX : v < INT16_MIN ? INT16_MIN : v;
}

// convert(f32, to=i32) under its default nearest_ties_positive rounding.
static int32_t round_i32(float v) {
  double r = floor((double)v + 0.5);
  return r >= 2147483648.0 ? INT32_MAX : r < -2147483648.0 ? INT32_MIN : (int32_t)r;
}

static void smooth_ref(const float *x, int n, float *y) {
  memcpy(y, x, 16 * sizeof(float));
  for (int k = 0; k < n; ++k)
    for (int i = 0; i < 16; ++i)
      y[i] = y[i] * 0.5f + x[i];
}

static void branch_ref(const float *x, int n, float *y) {
  memcpy(y, x, 16 * sizeof(float));
  for (int k = 0; k < n; ++k) {
    int mode = y[0] > 10.0f ? 0 : y[1] < 0.0f ? 1 : 2;
    for (int i = 0; i < 16; ++i)
      y[i] = (mode == 0 ? y[i] * 0.5f : mode == 1 ? -y[i] : y[i] + 1.0f) - 0.25f;
  }
}

int main(void) {
  for (int round = 0; round < 50; ++round) {
    float x[16], got[16], want[16];
    for (int i = 0; i < 16; ++i)
      x[i] = real();
    int n = round % 13;
    ondrix_tc_smooth(x, n, got);
    smooth_ref(x, n, want);
    if (!same(got, want, sizeof(got)))
      return fail("smooth", round);
    ondrix_tc_branch(x, n * 5, got);
    branch_ref(x, n * 5, want);
    if (!same(got, want, sizeof(got)))
      return fail("branch", round);
  }
  // Millions of iterations: a buffer allocated per iteration would exhaust the stack.
  {
    float x[16], got[16], want[16];
    for (int i = 0; i < 16; ++i)
      x[i] = real();
    x[0] = 3.0f;
    ondrix_tc_branch(x, 3000000, got);
    branch_ref(x, 3000000, want);
    if (!same(got, want, sizeof(got)))
      return fail("branch long", 0);
  }

  for (int round = 0; round < 50; ++round) {
    int32_t x[8], got[8], want[8];
    for (int i = 0; i < 8; ++i)
      x[i] = want[i] = (int32_t)(next() >> 32);
    int n = round % 9;
    for (int k = 0; k < n; ++k)
      for (int i = 0; i < 8; ++i)
        want[i] = (int32_t)(((uint32_t)want[i] + (uint32_t)x[i]) * 3u);
    ondrix_tc_self(x, n, got);
    if (!same(got, want, sizeof(got)))
      return fail("self", round);
  }

  for (int round = 0; round < 50; ++round) {
    int64_t x[4], got[4], want[4];
    for (int i = 0; i < 4; ++i)
      x[i] = want[i] = (int64_t)next();
    x[0] = want[0] = (int64_t)(next() % 1000) - 500;
    int64_t limit = (int64_t)(next() % (1ull << 40));
    while (want[0] < limit)
      for (int i = 0; i < 4; ++i)
        want[i] = (int64_t)((uint64_t)want[i] * 2u + 1u);
    ondrix_tc_while(x, limit, got);
    if (!same(got, want, sizeof(got)))
      return fail("while", round);
  }

  for (int round = 0; round < 50; ++round) {
    int64_t x[4], w[4], y[4], z[4], got[4], want[4];
    for (int i = 0; i < 4; ++i)
      x[i] = y[i] = (int64_t)next(), w[i] = z[i] = (int64_t)next();
    int n = round % 7;
    for (int k = 0; k < n; ++k)
      for (int i = 0; i < 4; ++i) {
        int64_t t = y[i];
        y[i] = z[i];
        z[i] = (int64_t)((uint64_t)t * 2u);
      }
    for (int i = 0; i < 4; ++i)
      want[i] = (int64_t)((uint64_t)y[i] - (uint64_t)z[i]);
    ondrix_tc_swap(x, w, n, got);
    if (!same(got, want, sizeof(got)))
      return fail("swap", round);
  }

  for (int round = 0; round < 50; ++round) {
    float x[4], w[4], y[4], s[4], got[4], want[4];
    for (int i = 0; i < 4; ++i)
      x[i] = y[i] = real(), w[i] = s[i] = real();
    int n = round % 6;
    for (int k = 0; k < n; ++k) {
      float z[4], c[4], v = y[2];
      int first = y[0] > 0.0f;
      for (int i = 0; i < 4; ++i)
        z[i] = y[i], c[i] = first ? y[i] : w[i];
      for (int i = 0; i < 4; ++i) {
        y[i] = y[i] * 0.5f - 1.0f;
        s[i] = s[i] * 0.25f + z[i] + y[i] + c[i];
        y[i] = y[i] + v;
      }
    }
    for (int i = 0; i < 4; ++i)
      want[i] = s[i] + y[i];
    ondrix_tc_alias(x, w, n, got);
    if (!same(got, want, sizeof(got)))
      return fail("alias", round);
  }

  for (int round = 0; round < 50; ++round) {
    float x[4], got[4], want[4];
    for (int i = 0; i < 4; ++i)
      x[i] = want[i] = real();
    int n = round % 5;
    for (int k = 0; k < n; ++k)
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j)
          want[i] = want[i] * 0.5f + x[i];
        want[i] = want[i] - 1.0f;
      }
    ondrix_tc_nest(x, n, got);
    if (!same(got, want, sizeof(got)))
      return fail("nest", round);
  }

  for (int round = 0; round < 200; ++round) {
    int16_t x[8], d[8], got[8], want[8];
    for (int i = 0; i < 8; ++i)
      x[i] = want[i] = (int16_t)(next() >> 48), d[i] = (int16_t)(next() >> 50);
    int n = round % 11;
    for (int k = 0; k < n; ++k) {
      int down = want[0] > x[0];
      for (int i = 0; i < 8; ++i)
        want[i] = sat16(down ? want[i] - d[i] : want[i] + d[i]);
    }
    ondrix_tc_q15(x, d, n, got);
    if (!same(got, want, sizeof(got)))
      return fail("q15", round);
  }

  for (int round = 0; round < 50; ++round) {
    int32_t x[8], y[8];
    for (int i = 0; i < 8; ++i)
      x[i] = y[i] = (int32_t)(next() >> 32) >> (round % 32);
    int n = round % 8;
    int64_t total = 0;
    for (int k = 0; k < n; ++k) {
      for (int i = 2; i < 6; ++i)
        total += y[i];
      for (int i = 0; i < 8; ++i)
        y[i] = (int32_t)((uint32_t)round_i32((float)y[i] * 0.75f) + 1u);
    }
    if (ondrix_tc_read(x, n) != total + y[7])
      return fail("read", round);
  }

  for (int round = 0; round < 50; ++round) {
    float x[4], y[4], w[4], got[4], want[4];
    for (int i = 0; i < 4; ++i)
      x[i] = real(), y[i] = w[i] = x[i] * 2.0f;
    int n = round % 6;
    for (int k = 0; k < n; ++k)
      for (int i = 0; i < 4; ++i) {
        float z = y[i];
        y[i] = (y[i] * 0.5f + w[i]) + z;
      }
    for (int i = 0; i < 4; ++i)
      want[i] = y[i] + w[i];
    ondrix_tc_outer(x, n, got);
    if (!same(got, want, sizeof(got)))
      return fail("outer", round);
  }

  for (int round = 0; round < 50; ++round) {
    int32_t x[4], got[4], want[4];
    for (int i = 0; i < 4; ++i)
      x[i] = want[i] = (int32_t)(next() % 400) - 100;
    int n = round % 9;
    for (int k = 0; k < n; ++k)
      for (int j = 0; j < 3; ++j) {
        int high = want[0] > 100;
        for (int i = 0; i < 4; ++i)
          want[i] = want[i] - (high ? 7 : 0) + 1;
      }
    ondrix_tc_mix(x, n, got);
    if (!same(got, want, sizeof(got)))
      return fail("mix", round);
  }

  for (int round = 0; round < 50; ++round) {
    float x[4], w[4], y[4], got[4];
    for (int i = 0; i < 4; ++i)
      x[i] = y[i] = real(), w[i] = real();
    int n = round % 6;
    for (int k = 0; k < n; ++k) {
      float c[4], d[4];
      int first = y[0] > 0.0f;
      for (int i = 0; i < 4; ++i)
        c[i] = first ? y[i] : w[i], d[i] = c[i] * 2.0f;
      int keep = d[1] > 1.0f;
      for (int i = 0; i < 4; ++i)
        y[i] = keep ? d[i] : c[i];
    }
    ondrix_tc_pick(x, w, n, got);
    if (!same(got, y, sizeof(got)))
      return fail("pick", round);
  }

  for (int round = 0; round < 50; ++round) {
    float x[4], y[4], s[4], z[4], got[4], want[4];
    for (int i = 0; i < 4; ++i)
      x[i] = y[i] = s[i] = real() * 4.0f, z[i] = x[i] * 2.0f;
    int n = round % 8;
    for (int k = 0; k < n; ++k) {
      for (int i = 0; i < 4; ++i) {
        float t = y[i] * 2.0f;
        y[i] = t + 1.0f;
        s[i] = s[i] + t;
      }
      if (s[0] > 50.0f)
        memcpy(y, z, sizeof(y));
    }
    for (int i = 0; i < 4; ++i)
      want[i] = s[i] + y[i];
    ondrix_tc_named(x, n, got);
    if (!same(got, want, sizeof(got)))
      return fail("named", round);
  }
  return 0;
}

#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

// Python range semantics in a wider type, so no reference induction wraps.
static uint32_t hash32(uint32_t acc, int64_t start, int64_t stop, int64_t step) {
  for (int64_t k = start; step > 0 ? k < stop : k > stop; k += step)
    acc = acc * 31u + (uint32_t)k;
  return acc;
}

static uint64_t hash64(uint64_t acc, __int128 start, __int128 stop, __int128 step) {
  for (__int128 k = start; step > 0 ? k < stop : k > stop; k += step)
    acc = acc * 31u + (uint64_t)k;
  return acc;
}

static int ranges(void) {
  static const int32_t small32[][2] = {{0, 0},
                                       {0, 10},
                                       {10, 0},
                                       {-10, 17},
                                       {17, -10},
                                       {INT32_MAX - 5, INT32_MAX},
                                       {INT32_MAX - 1, INT32_MAX},
                                       {INT32_MIN, INT32_MIN + 7},
                                       {INT32_MIN + 1, INT32_MIN}};
  static const int32_t wide32[][2] = {{INT32_MIN, INT32_MAX},
                                      {INT32_MAX, INT32_MIN},
                                      {INT32_MIN, INT32_MIN + 1},
                                      {INT32_MAX - 1, INT32_MAX},
                                      {-5, 5},
                                      {0, 0}};
  for (unsigned i = 0; i < sizeof(small32) / sizeof(small32[0]); ++i) {
    for (unsigned w = 0; w < sizeof(wide32) / sizeof(wide32[0]); ++w) {
      int64_t a = small32[i][0], b = small32[i][1], c = wide32[w][0], d = wide32[w][1];
      uint32_t acc = hash32(0, a, b, 1);
      acc = hash32(acc, a, b, 3);
      acc = hash32(acc, b, a, -1);
      acc = hash32(acc, b, a, -3);
      acc = hash32(acc, c, d, 1000003);
      acc = hash32(acc, d, c, -999983);
      acc = hash32(acc, c, d, INT32_MAX);
      acc = hash32(acc, d, c, INT32_MIN);
      int32_t got = ondrix_ranges_i32((int32_t)a, (int32_t)b, (int32_t)c, (int32_t)d);
      if ((uint32_t)got != acc) {
        fprintf(stderr, "i32 ranges %u/%u\n", i, w);
        return 1;
      }
    }
  }
  static const int64_t small64[][2] = {{0, 0},
                                       {0, 10},
                                       {10, 0},
                                       {-10, 17},
                                       {17, -10},
                                       {INT64_MAX - 5, INT64_MAX},
                                       {INT64_MIN, INT64_MIN + 7},
                                       {INT64_MIN + 1, INT64_MIN}};
  static const int64_t wide64[][2] = {{INT64_MIN, INT64_MAX},
                                      {INT64_MAX, INT64_MIN},
                                      {INT64_MIN, INT64_MIN + 1},
                                      {INT64_MAX - 1, INT64_MAX},
                                      {-5, 5}};
  for (unsigned i = 0; i < sizeof(small64) / sizeof(small64[0]); ++i) {
    for (unsigned w = 0; w < sizeof(wide64) / sizeof(wide64[0]); ++w) {
      __int128 a = small64[i][0], b = small64[i][1], c = wide64[w][0], d = wide64[w][1];
      uint64_t acc = hash64(0, a, b, 1);
      acc = hash64(acc, a, b, 3);
      acc = hash64(acc, b, a, -1);
      acc = hash64(acc, b, a, -3);
      acc = hash64(acc, c, d, INT64_C(1000000000000000003));
      acc = hash64(acc, d, c, -INT64_C(999999999999999989));
      acc = hash64(acc, c, d, INT64_MAX);
      acc = hash64(acc, d, c, INT64_MIN);
      int64_t got = ondrix_ranges_i64(small64[i][0], small64[i][1], wide64[w][0], wide64[w][1]);
      if ((uint64_t)got != acc) {
        fprintf(stderr, "i64 ranges %u/%u\n", i, w);
        return 1;
      }
    }
  }
  return 0;
}

static uint64_t state = 0x9e3779b97f4a7c15u;

// Signs and exponents spread over 2^-20..2^20 make every summation order observable.
static float sample(void) {
  state ^= state << 13;
  state ^= state >> 7;
  state ^= state << 17;
  float mantissa = 1.0f + (float)(state & 0xffff) / 65536.0f;
  int exponent = (int)((state >> 16) % 41) - 20;
  float value = mantissa;
  for (; exponent > 0; --exponent)
    value *= 2.0f;
  for (; exponent < 0; ++exponent)
    value *= 0.5f;
  return (state >> 40) & 1 ? -value : value;
}

static int same(float a, float b) { return memcmp(&a, &b, sizeof(a)) == 0; }

static int reads(void) {
  float x[40], y[40], fixed[8];
  for (unsigned i = 0; i < 40; ++i) {
    x[i] = sample();
    y[i] = sample();
  }
  for (unsigned n = 0; n <= 33; ++n) {
    float total = 0.0f;
    for (unsigned i = 0; i < n; ++i)
      total = total + x[i] * x[i];
    if (!same(ondrix_energy(x, n), total)) {
      fprintf(stderr, "energy %u\n", n);
      return 1;
    }
  }
  for (unsigned n = 0; n <= 6; ++n) {
    for (unsigned m = 0; m <= 7; ++m) {
      float s = 0.0f;
      for (int i = (int)n - 1; i > -1; --i) {
        float row = 0.0f;
        for (unsigned j = 1; j < m; j += 2)
          row = row + x[i] * y[j - 1];
        s = s * 0.5f + row;
      }
      if (!same(ondrix_mix(x, n, y, m), s)) {
        fprintf(stderr, "mix %u %u\n", n, m);
        return 1;
      }
    }
  }
  for (unsigned i = 0; i < 8; ++i)
    fixed[i] = sample();
  float s = 0.0f;
  for (unsigned i = 0; i < 8; i += 3)
    s = s + fixed[i];
  for (unsigned i = 2; i < 8; ++i)
    s = s - fixed[i];
  if (!same(ondrix_tensor_sum(fixed), s) || !same(ondrix_at(x, 4, 3, 0), x[3] + x[0]) ||
      !same(ondrix_overrun(x, 0), 0.0f)) {
    fprintf(stderr, "static, direct or empty reads\n");
    return 1;
  }
  return 0;
}

static int traps(unsigned mode) {
  float x[8] = {0};
  pid_t child = fork();
  if (child == 0) {
    switch (mode) {
    case 0:
      ondrix_at(x, 4, -1, 0);
      break;
    case 1:
      ondrix_at(x, 4, 4, 0);
      break;
    case 2:
      ondrix_at(x, 4, INT64_C(1) << 32, 0);
      break;
    case 3:
      ondrix_at(x, 4, 0, -1);
      break;
    case 4:
      ondrix_at(x, 4, 0, 4);
      break;
    case 5:
      ondrix_overrun(x, 3);
      break;
    case 6:
      ondrix_at(x, 4, 3, 3);
      break;
    }
    _exit(0);
  }
  int status;
  return child > 0 && waitpid(child, &status, 0) == child && WIFSIGNALED(status) &&
         (WTERMSIG(status) == SIGABRT || WTERMSIG(status) == SIGILL);
}

int main(void) {
  if (ranges() || reads())
    return 1;
  // Mode 6 reads in bounds: the trap check must also be able to fail.
  for (unsigned mode = 0; mode < 7; ++mode)
    if (traps(mode) != (mode < 6)) {
      fprintf(stderr, "subscript trap mode %u\n", mode);
      return 1;
    }
  return 0;
}

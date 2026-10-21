#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#define N 200
#define TAPS 16

static uint64_t state = 0x2545f4914f6cdd1du;

static uint64_t next(void) {
  state ^= state << 13;
  state ^= state >> 7;
  state ^= state << 17;
  return state;
}

static int fail(const char *what, unsigned n) {
  fprintf(stderr, "%s %u\n", what, n);
  return 1;
}

static int64_t saturate(int64_t value, int bits) {
  int64_t top = (INT64_C(1) << (bits - 1)) - 1;
  return value > top ? top : value < -top - 1 ? -top - 1 : value;
}

// The 40-bit accumulator exported to q15: nearest, ties toward +infinity, saturated.
static int16_t export15(int64_t acc) {
  int64_t floor = (acc - (acc & 32767)) / 32768;
  if ((acc & 32767) >= 16384)
    ++floor;
  return (int16_t)saturate(floor, 16);
}

static int16_t dot15(const int16_t *x, const int16_t *h, unsigned n) {
  int64_t acc = 0;
  for (unsigned k = 0; k < n; ++k)
    acc = saturate(acc + (int64_t)x[k] * h[k], 40);
  return export15(acc);
}

// The off contract: one ordered fold seeded by the first product; empty is +0.0.
static float dot32(const float *x, const float *w, unsigned n) {
  float sum = n ? -0.0f : 0.0f;
  for (unsigned i = 0; i < n; ++i)
    sum = sum + x[i] * w[i];
  return sum;
}

static int same(float a, float b) { return memcmp(&a, &b, sizeof(a)) == 0; }

static float page[16] = {1, 2, 3, 4, 5, 6, 7, 8};
static int64_t from, to;
static void slicePage(void) { printf("%g\n", ondrix_sl_trap(page, 4, from, to)); }
// Constant bounds 2:10 over a dynamic base of nine elements.
static void shortWindow(void) { printf("%g\n", ondrix_sl_win(page, 9, page, 0, page + 9, 0)); }

// Runs `call` in a child; true when it aborted after printing `message`.
static int dies(void (*call)(void), const char *message) {
  int pipefd[2];
  if (pipe(pipefd) != 0)
    return 0;
  pid_t child = fork();
  if (child == 0) {
    dup2(pipefd[1], STDOUT_FILENO);
    close(pipefd[0]);
    setvbuf(stdout, NULL, _IONBF, 0);
    call();
    _exit(0);
  }
  close(pipefd[1]);
  char output[256] = {0};
  size_t count = 0;
  for (ssize_t got = 1; got > 0 && count < sizeof(output) - 1; count += (size_t)got)
    got = read(pipefd[0], output + count, sizeof(output) - 1 - count);
  close(pipefd[0]);
  int status = 0;
  return child > 0 && waitpid(child, &status, 0) == child && WIFSIGNALED(status) &&
         WTERMSIG(status) == SIGABRT && strstr(output, message);
}

int main(void) {
  static int16_t x[N + TAPS], h[TAPS], y[N + 1], d[TAPS + 3];
  static float fx[N + TAPS + 10], fh[TAPS], fy[N + 1];
  for (unsigned i = 0; i < N + TAPS; ++i)
    x[i] = (int16_t)(next() >> 48);
  for (unsigned i = 0; i < N + TAPS + 10; ++i)
    fx[i] = (float)(int32_t)(next() >> 40) / 4096.0f;
  for (unsigned i = 0; i < TAPS; ++i) {
    h[i] = (int16_t)(next() >> 48);
    fh[i] = (float)(int32_t)(next() >> 40) / 4096.0f;
  }

  int rezeroed = 0;
  for (unsigned n = 0; n <= N; n += 1 + n / 4) {
    for (unsigned taps = 0; taps <= TAPS; taps += 4) {
      y[n] = 0x5a5a;
      ondrix_sl_qfir(x, n + taps, h, taps, y, n);
      for (unsigned i = 0; i < n; ++i)
        if (y[i] != dot15(x + i, h, taps))
          return fail("qfir", n * 100 + taps);
      if (y[n] != 0x5a5a)
        return fail("qfir guard", n);

      float got = ondrix_sl_win(fx, n + taps + 10, fh, taps, fy, n);
      for (unsigned i = 0; i < n; ++i)
        if (!same(fy[i], dot32(fx + i, fh, taps)))
          return fail("win", n * 100 + taps);
      if (!same(got, sqrtf(dot32(fx + 2, fx + 2, 8) / 8.0f) + dot32(fy, fy, n)))
        return fail("win result", n * 100 + taps);
    }

    // Without re-zeroing, an element below `i` would keep the previous write.
    ondrix_sl_scratch(fx, n, fy, n);
    float kept[4] = {0};
    for (unsigned i = 0; i < n; ++i) {
      float t[4];
      for (unsigned k = 0; k < 4; ++k) {
        t[k] = k >= i ? fx[i] : 0.0f;
        kept[k] = k >= i ? fx[i] : kept[k];
      }
      if (!same(fy[i], t[0] + t[1] + t[2] + t[3]))
        return fail("scratch", i);
      rezeroed |= !same(fy[i], kept[0] + kept[1] + kept[2] + kept[3]);
    }
  }
  if (!rezeroed)
    return fail("scratch witness", 0);

  for (unsigned length = 0; length <= 8; ++length) {
    if (ondrix_sl_taps(x, length, h, TAPS) != dot15(x, h, length))
      return fail("taps", length);
    float sum = 0.0f;
    for (unsigned i = length; i < 8; ++i)
      sum = sum + fx[i];
    if (!same(ondrix_sl_tail(fx, length), sum))
      return fail("tail", length);
  }

  // The slice is bound before the shift: a view reads the shifted line, a copy would not.
  memset(d, 0, sizeof d);
  int shifted = 0;
  for (unsigned n = 0; n < N; ++n) {
    int16_t line[TAPS], stale[TAPS];
    for (unsigned k = 0; k < TAPS; ++k) {
      line[k] = k <= n ? x[n - k] : 0;
      stale[k] = k + 1 <= n ? x[n - 1 - k] : 0;
    }
    int16_t want = dot15(line, h, TAPS);
    shifted |= want != dot15(stale, h, TAPS);
    if (ondrix_sl_delay(d, TAPS + 3, h, TAPS, x[n]) != want)
      return fail("delay", n);
  }
  if (!shifted)
    return fail("delay witness", 0);

  static const int64_t bounds[][2] = {{-1, 2}, {3, 2}, {0, 5}, {0, (INT64_C(1) << 32) + 2}};
  for (unsigned i = 0; i < 4; ++i) {
    from = bounds[i][0], to = bounds[i][1];
    if (!dies(slicePage, "slice out of bounds"))
      return fail("slice trap", i);
  }
  if (!dies(shortWindow, "slice out of bounds"))
    return fail("slice trap", 4);
  // A slice inside the bounds: the trap check must also be able to fail.
  from = 1, to = 3;
  if (dies(slicePage, "slice out of bounds") || !same(ondrix_sl_trap(page, 4, 1, 3), 13.0f))
    return fail("slice control", 0);
  if (!same(ondrix_sl_trap(page, 4, 4, 4), 0.0f))
    return fail("empty slices", 0);
  return 0;
}

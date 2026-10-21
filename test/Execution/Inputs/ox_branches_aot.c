#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#define N 64

static uint64_t state = 0x9e3779b97f4a7c15u;

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

static int same(float a, float b) { return memcmp(&a, &b, sizeof(a)) == 0; }

static int64_t gcd(int64_t a, int64_t b) {
  a = a < 0 ? -a : a;
  b = b < 0 ? -b : b;
  while (b) {
    int64_t t = a % b;
    a = b;
    b = t;
  }
  return a;
}

static float page[4] = {1, 2, 3, 4};
static void wideWindow(void) { ondrix_br_window(page, 1); }
static void zeroRatio(void) { printf("%d\n", ondrix_br_ratio(7, 0)); }

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
  static float x[N], y[N + 1];
  static int16_t q[N], qy[N + 1];
  int low = 0, high = 0;
  for (unsigned n = 0; n <= N; n += 1 + n / 3) {
    for (unsigned i = 0; i < n; ++i) {
      x[i] = (float)((int32_t)(next() >> 40) - (1 << 23)) / 2048.0f;
      q[i] = (int16_t)(next() >> 48);
    }
    // A clipped-low value passes the elif test, so an elif read as a second if counts it twice.
    y[n] = qy[n] = 77;
    int clipped = 0;
    for (unsigned i = 0; i < n; ++i)
      clipped += x[i] < -1000.0f ? 1 : x[i] <= 1000.0f ? 100 : 0;
    for (unsigned i = 0; i < n; ++i)
      low |= x[i] < -1000.0f, high |= x[i] > 1000.0f;
    if (ondrix_br_clip(x, n, y, n, -1000.0f, 1000.0f) != clipped)
      return fail("clip count", n);
    for (unsigned i = 0; i < n; ++i)
      if (!same(y[i], x[i] < -1000.0f ? -1000.0f : x[i] > 1000.0f ? 1000.0f : x[i]))
        return fail("clip", i);

    // The empty case reads no x[0]: an eager arm would trap.
    int64_t at = -1;
    for (unsigned i = 0; i < n; ++i)
      if (at < 0 || q[i] > q[at])
        at = i;
    if (ondrix_br_peak(q, n, qy, n) != at)
      return fail("peak", n);
    for (unsigned i = 0, best = 0; i < n; ++i) {
      best = q[i] > q[best] ? i : best;
      if (qy[i] != q[best])
        return fail("peak running", i);
    }

    // The last element takes the else body; the then body would read x[n].
    ondrix_br_diff(x, n, y, n);
    for (unsigned i = 0; i < n; ++i)
      if (!same(y[i], i + 1 < n ? x[i + 1] - x[i] : 0.0f - x[i]))
        return fail("diff", i);
    if (y[n] != 77 || qy[n] != 77)
      return fail("guard", n);
  }

  if (!low || !high)
    return fail("clip witness", 0);

  for (int64_t a = -40; a <= 40; a += 3)
    for (int64_t b = -60; b <= 60; b += 7)
      if (ondrix_br_gcd(a, b) != gcd(a, b))
        return fail("gcd", a * 1000 + b);
  if (ondrix_br_gcd(0, 0) != 0 || ondrix_br_gcd(9, 0) != 9)
    return fail("gcd empty", 0);

  for (int32_t a = -9; a <= 9; ++a)
    for (int32_t b = -4; b <= 4; ++b)
      if (ondrix_br_ratio(a, b) != (b ? a / b : 0) * 4 + (a > 0) - (a < 0))
        return fail("ratio", a * 100 + b);
  if (dies(zeroRatio, "integer division"))
    return fail("ratio witness", 0);

  float in[8], out[8];
  for (unsigned i = 0; i < 8; ++i)
    in[i] = (float)((int32_t)(next() >> 40) - (1 << 23)) / 512.0f;
  for (uint8_t c = 0; c < 2; ++c) {
    ondrix_br_mix(in, c, out);
    for (unsigned i = 0; i < 8; ++i)
      if (!same(out[i], c ? in[i] * 2.0f : in[i] - 1.0f))
        return fail("mix", c * 10 + i);
  }

  // A constant bound assigned in a branch is unknown after it: the check stays.
  ondrix_br_window(page, 0);
  if (!same(page[0], 30.0f))
    return fail("window", 0);
  if (!dies(wideWindow, "slice out of bounds"))
    return fail("window trap", 0);
  return 0;
}

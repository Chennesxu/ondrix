#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#define N 300
#define TAPS 24
#define GUARD 4

static int fail(const char *what, unsigned n) {
  fprintf(stderr, "%s %u\n", what, n);
  return 1;
}

// Runs `call` in a child; true when it aborted after printing `message`.
static int refuses(void (*call)(void), const char *message) {
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
  if (child < 0 || waitpid(child, &status, 0) != child || !WIFSIGNALED(status) ||
      WTERMSIG(status) != SIGABRT || !strstr(output, message)) {
    fprintf(stderr, "%s: not refused ('%s')\n", message, output);
    return 0;
  }
  return 1;
}

#ifdef CHECKED
static float page[16];
static int16_t state16[16];
static void scaleOverInput(void) { ondrix_out_scale(page, 8, page + 4, 8, 2.0f); }
static void delayOverTaps(void) { ondrix_out_delay(state16, 8, state16 + 7, 8, 1); }

int main(void) {
  float x[4] = {1, 2, 3, 4}, y[4];
  ondrix_out_scale(x, 4, y, 4, 0.5f);
  if (y[0] != 0.5f || y[3] != 2.0f)
    return fail("checked scale", 0);
  return !refuses(scaleOverInput, "ondrix_out_scale: two buffers overlap") ||
         !refuses(delayOverTaps, "ondrix_out_delay: two buffers overlap");
}
#else
static uint64_t state = 0x2545f4914f6cdd1du;

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

// The 40-bit accumulator exported to q15: nearest, ties toward +infinity, saturated.
static int16_t export15(int64_t acc) {
  int64_t floor = (acc - (acc & 32767)) / 32768;
  if ((acc & 32767) >= 16384)
    ++floor;
  return (int16_t)saturate(floor, 16);
}

// A write evaluates its value first: a zero divisor traps before the bad index.
static void zeroDivisorAtBadIndex(void) {
  int16_t y[1];
  ondrix_out_order(y, 1, -1, 1, 0);
}

static int traps(unsigned mode) {
  int32_t y[4] = {0};
  float x[4] = {0};
  int16_t d[1] = {0}, h[1] = {0};
  pid_t child = fork();
  if (child == 0) {
    switch (mode) {
    case 0:
      ondrix_out_put(y, 4, -1, 7);
      break;
    case 1:
      ondrix_out_put(y, 4, 4, 7);
      break;
    case 2:
      ondrix_out_put(y, 4, INT64_C(1) << 32, 7);
      break;
    case 3:
      ondrix_out_delay(d, 0, h, 1, 7);
      break;
    case 4:
      ondrix_out_scale(x, 3, (float *)y, 4, 1.0f);
      break;
    case 5:
      ondrix_out_put(y, 4, 3, 7);
      break;
    }
    _exit(0);
  }
  int status;
  return child > 0 && waitpid(child, &status, 0) == child && WIFSIGNALED(status) &&
         (WTERMSIG(status) == SIGABRT || WTERMSIG(status) == SIGILL);
}

int main(void) {
  static int16_t x[N + TAPS], h[TAPS], y[N + GUARD], d[TAPS];
  static float fx[N], fy[N + GUARD], before[N];
  for (unsigned i = 0; i < N + TAPS; ++i)
    x[i] = (int16_t)(next() >> 48);
  for (unsigned i = 0; i < TAPS; ++i)
    h[i] = (int16_t)(next() >> 48);
  for (unsigned i = 0; i < N; ++i)
    fx[i] = (float)(int32_t)(next() >> 40) / 4096.0f;

  for (unsigned n = 0; n <= N; n += 1 + n / 8) {
    for (unsigned taps = 0; taps <= TAPS; taps += 6) {
      for (unsigned i = 0; i < N + GUARD; ++i)
        y[i] = (int16_t)0x5a5a;
      ondrix_out_fir(x, n + (taps ? taps - 1 : 0), h, taps, y, n);
      for (unsigned i = 0; i < n; ++i) {
        int64_t acc = 0;
        for (unsigned k = 0; k < taps; ++k)
          acc = saturate(acc + (int64_t)x[i + k] * h[k], 40);
        if (y[i] != export15(acc))
          return fail("fir", n * 100 + taps);
      }
      for (unsigned i = n; i < n + GUARD; ++i)
        if (y[i] != (int16_t)0x5a5a)
          return fail("fir guard", n);
    }

    for (unsigned i = 0; i < N + GUARD; ++i)
      fy[i] = -0.0f;
    ondrix_out_scale(fx, N, fy, n, -1.5f);
    for (unsigned i = 0; i < n + GUARD; ++i) {
      float want = i < n ? -1.5f * fx[i] : -0.0f;
      if (memcmp(&fy[i], &want, sizeof want))
        return fail("scale", i);
    }

    memcpy(fy, fx, sizeof fx);
    memcpy(before, fx, sizeof fx);
    ondrix_out_prefix(fy, n);
    float sum = 0.0f;
    int ownWrites = 0;
    for (unsigned i = 0; i < N; ++i) {
      float want = i < n ? (i ? sum + before[i] : before[0]) : before[i];
      if (i > 1 && i < n)
        ownWrites |= want != before[i] + before[i - 1];
      sum = want;
      if (memcmp(&fy[i], &want, sizeof want))
        return fail("prefix", i);
    }
    if (n > 8 && !ownWrites)
      return fail("prefix witness", n);
  }

  // A delay line the kernel shifts in place: equal to direct convolution.
  for (unsigned length = 1; length <= TAPS; length += 7) {
    memset(d, 0, sizeof d);
    for (unsigned n = 0; n < N; ++n) {
      int64_t acc = 0;
      for (unsigned k = 0; k < length; ++k)
        acc = saturate(acc + (int64_t)(k <= n ? x[n - k] : 0) * h[k], 40);
      if (ondrix_out_delay(d, length, h, TAPS, x[n]) != export15(acc))
        return fail("delay", n);
      for (unsigned k = 0; k < length; ++k)
        if (d[k] != (k <= n ? x[n - k] : 0))
          return fail("delay state", n);
    }
  }

  int32_t q[4] = {1, 2, 3, 4};
  ondrix_out_put(q, 4, 2, INT32_MIN);
  if (q[0] != 1 || q[1] != 2 || q[2] != INT32_MIN || q[3] != 4)
    return fail("put", 0);
  // Mode 5 writes in bounds: the trap check must also be able to fail.
  for (unsigned mode = 0; mode < 6; ++mode)
    if (traps(mode) != (mode < 5))
      return fail("trap mode", mode);
  if (!refuses(zeroDivisorAtBadIndex, "the divisor is not positive"))
    return fail("evaluation order", 0);
  return 0;
}
#endif

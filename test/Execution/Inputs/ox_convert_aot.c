#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#define N 40

static uint64_t state = 0xbb67ae8584caa73bu;

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

static float bits(uint32_t u) {
  float f;
  memcpy(&f, &u, sizeof(f));
  return f;
}

// Rounds in double, where every f32 plus one half is exact below 2^52 and
// larger f32 values are already integers; then saturates at `width` bits.
static int64_t round_ref(float v, int mode, int width) {
  if (v != v)
    return 0;
  double d = v;
  double r = mode == 0 ? trunc(d) : mode == 1 ? floor(d) : mode == 2 ? rint(d) : floor(d + 0.5);
  double rail = ldexp(1.0, width - 1);
  if (r >= rail)
    return width == 32 ? INT32_MAX : INT64_MAX;
  if (r < -rail)
    return width == 32 ? INT32_MIN : INT64_MIN;
  return (int64_t)r;
}

static float in[N];
static int32_t start;
static int64_t stop;
static void tail(void) { printf("%f\n", ondrix_cv_tail(in, N, start, stop)); }

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
  for (unsigned i = 0; i < N; ++i)
    in[i] = (float)((int32_t)(next() >> 40) - (1 << 23)) / 2048.0f;

  for (unsigned n = 1; n <= N; ++n) {
    float s = 0.0f;
    for (unsigned i = 0; i < n; ++i)
      s += in[i];
    if (!same(ondrix_cv_mean(in, n), s / (float)n))
      return fail("mean", n);
  }

  // Ties of both signs and parities, both rails and their neighbours, NaN,
  // infinities, signed zeros and subnormals, then random bit patterns.
  static const float edges[] = {
      0.5f,          -0.5f,        1.5f,          -1.5f,          2.5f,           -2.5f,
      0.49999997f,   -0.49999997f, 8388607.5f,    -8388607.5f,    0.0f,           -0.0f,
      1e-45f,        -1e-45f,      2147483520.0f, 2147483648.0f,  -2147483648.0f, -2147483904.0f,
      9.2233715e18f, 9.223372e18f, -9.223372e18f, -9.2233731e18f, INFINITY,       -INFINITY,
      NAN,           -NAN,         3.75f,         -3.25f};
  unsigned edgeCount = sizeof(edges) / sizeof(edges[0]);
  for (unsigned i = 0; i < edgeCount + 200000; ++i) {
    float v = i < edgeCount ? edges[i] : bits((uint32_t)(next() >> 32));
    for (int mode = 0; mode < 4; ++mode) {
      if (ondrix_cv_round(v, mode) != round_ref(v, mode, 32))
        return fail("round", (long)i * 10 + mode);
      if (ondrix_cv_round64(v, mode) != round_ref(v, mode, 64))
        return fail("round64", (long)i * 10 + mode);
    }
  }
  // Constant operands past each rail fold through the same clamp.
  if (ondrix_cv_round(0.0f, 4) != INT32_MAX || ondrix_cv_round(0.0f, 5) != INT32_MIN ||
      ondrix_cv_round64(0.0f, 4) != INT64_MAX || ondrix_cv_round64(0.0f, 5) != INT64_MIN)
    return fail("round constant", 0);

  // Each rail, its neighbours past it, and values beyond 32 bits of both signs.
  static const int64_t wide[] = {0,          1,           -1,          INT32_MAX,    INT32_MIN,
                                 2147483648, -2147483649, 68719476736, -68719476736, INT64_MAX,
                                 INT64_MIN};
  for (unsigned i = 0; i < sizeof(wide) / sizeof(wide[0]) + 1000; ++i) {
    int64_t v = i < sizeof(wide) / sizeof(wide[0]) ? wide[i] : (int64_t)next();
    int32_t saturated = v > INT32_MAX ? INT32_MAX : v < INT32_MIN ? INT32_MIN : (int32_t)v;
    if (ondrix_cv_narrow(v, 0) != saturated || ondrix_cv_narrow(v, 1) != (int32_t)(uint32_t)v)
      return fail("narrow", i);
  }

  for (int32_t k = -3; k <= 3; ++k)
    for (unsigned n = 0; n <= 5; ++n)
      if (ondrix_cv_widen(in, n, k) != (int64_t)k * 65536 + ((int64_t)n - 3) * 2 + n)
        return fail("widen", k * 10 + (int)n);

  static const int32_t narrowFloats[] = {0, -7, 16777217, -16777219, INT32_MAX, INT32_MIN};
  static const int64_t wideFloats[] = {3, -16777217, 0x4000004000000001LL, INT64_MIN, INT64_MAX};
  float pair[2];
  for (unsigned i = 0; i < 6; ++i)
    for (unsigned j = 0; j < 5; ++j) {
      ondrix_cv_float(pair, 2, narrowFloats[i], wideFloats[j]);
      if (!same(pair[0], (float)narrowFloats[i]) || !same(pair[1], (float)wideFloats[j]))
        return fail("float", i * 10 + j);
    }

  for (start = 0; start <= N; ++start)
    for (stop = 0; start + stop <= N; ++stop) {
      float s = 0.0f;
      for (int i = start; i < N - stop; ++i)
        s += in[i];
      if (!same(ondrix_cv_tail(in, N, start, stop), s))
        return fail("tail", start * 100 + (long)stop);
    }
  // A negative start stays negative: x[-1] traps, where a zero extension would skip the loop.
  start = -1, stop = 0;
  if (!dies(tail, "index out of bounds"))
    return fail("tail start", 0);
  start = 0, stop = -1;
  if (!dies(tail, "index out of bounds"))
    return fail("tail stop", 0);
  return 0;
}

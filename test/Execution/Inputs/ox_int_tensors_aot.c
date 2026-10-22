#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#define N 40

static uint64_t state = 0x3c6ef372fe94f82bu;

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

// Nearest-even in double, where every f32 is exact; then saturated to i32.
static int32_t quant_ref(float v) {
  if (v != v)
    return 0;
  double r = rint((double)v);
  if (r >= 2147483648.0)
    return INT32_MAX;
  if (r < -2147483648.0)
    return INT32_MIN;
  return (int32_t)r;
}

static float xs[N];
static int32_t perm[N];
static float gathered[N];
static int64_t table[8], dividend[8], divisor[8], quotient[8];
static int32_t pick;
static void gather(void) { ondrix_it_gather(xs, N, perm, N, gathered, N); }
static void pick_call(void) { printf("%ld\n", (long)ondrix_it_pick(table, pick)); }
static void divide(void) { ondrix_it_div(dividend, divisor, quotient); }

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
  // Bins start nonzero: the kernel adds to the caller's contents.
  for (int round = 0; round < 200; ++round) {
    int32_t x[N], missed = 0;
    int64_t h[8], expected[8];
    unsigned n = (unsigned)round % (N + 1);
    for (unsigned i = 0; i < N; ++i)
      x[i] = (int32_t)(next() % 16) - 4;
    x[0] = round % 3 == 0 ? INT32_MIN : round % 3 == 1 ? INT32_MAX : x[0];
    for (unsigned i = 0; i < 8; ++i)
      h[i] = expected[i] = (int64_t)i * 1000 - 3;
    for (unsigned i = 0; i < n; ++i)
      if (x[i] >= 0 && x[i] < 8)
        ++expected[x[i]];
      else
        ++missed;
    if (ondrix_it_hist(x, n, h, 8) != missed || memcmp(h, expected, sizeof(h)) != 0)
      return fail("hist", round);
  }

  for (unsigned i = 0; i < N; ++i) {
    xs[i] = (float)((int32_t)(next() >> 40) - (1 << 23)) / 2048.0f;
    perm[i] = (int32_t)(next() % N);
  }
  gather();
  for (unsigned i = 0; i < N; ++i)
    if (!same(gathered[i], xs[perm[i]]))
      return fail("gather", i);
  // A negative i32 element stays negative as a subscript.
  perm[5] = -1;
  if (!dies(gather, "index out of bounds"))
    return fail("gather low", 0);
  perm[5] = N;
  if (!dies(gather, "index out of bounds"))
    return fail("gather high", 0);

  for (unsigned i = 0; i < 8; ++i)
    table[i] = (int64_t)next();
  table[0] = INT64_MIN, table[7] = 1;
  for (pick = 0; pick < 8; ++pick)
    if (ondrix_it_pick(table, pick) != (int64_t)((uint64_t)table[pick] - (uint64_t)table[7 - pick]))
      return fail("pick", pick);
  pick = 8;
  if (!dies(pick_call, "index out of bounds"))
    return fail("pick high", 0);
  pick = -1;
  if (!dies(pick_call, "index out of bounds"))
    return fail("pick low", 0);

  for (int round = 0; round < 1000; ++round) {
    int32_t a[8], b[8], out[8];
    for (unsigned i = 0; i < 8; ++i)
      a[i] = (int32_t)(next() >> 32), b[i] = (int32_t)(next() >> 32);
    a[0] = INT32_MIN, b[0] = -1;
    int32_t k = (int32_t)(next() >> 32);
    ondrix_it_wrap(a, b, k, out);
    for (unsigned i = 0; i < 8; ++i) {
      uint32_t sum = (uint32_t)a[i] * (uint32_t)b[i] + (uint32_t)k;
      if (out[i] != (int32_t)((0u - sum) * 3u - (uint32_t)a[i]))
        return fail("wrap", round * 10 + (long)i);
    }
  }

  for (int round = 0; round < 1000; ++round) {
    for (unsigned i = 0; i < 8; ++i) {
      dividend[i] = (int64_t)next() >> (next() % 64);
      do
        divisor[i] = (int64_t)next() >> (next() % 64);
      while (divisor[i] == 0 || divisor[i] == -1);
    }
    dividend[1] = INT64_MIN, divisor[2] = -1, dividend[3] = -7, divisor[3] = 2;
    divide();
    for (unsigned i = 0; i < 8; ++i)
      if (quotient[i] !=
          (int64_t)((uint64_t)(dividend[i] / divisor[i]) + (uint64_t)(dividend[i] / 7)))
        return fail("div", round * 10 + (long)i);
  }
  divisor[6] = 0;
  if (!dies(divide, "integer division by zero or overflow"))
    return fail("div zero", 0);
  divisor[6] = 3, dividend[4] = INT64_MIN, divisor[4] = -1;
  if (!dies(divide, "integer division by zero or overflow"))
    return fail("div overflow", 0);

  // Ties of both parities and signs, both rails, NaN and infinities, then random bits.
  static const float edges[] = {
      0.5f,          -0.5f,         1.5f,           -1.5f,          2.5f,     -2.5f,
      2147483520.0f, 2147483648.0f, -2147483648.0f, -2147483904.0f, INFINITY, -INFINITY,
      NAN,           -0.0f,         1e-45f,         8388607.5f};
  for (int round = 0; round < 20000; ++round) {
    float v[16];
    int32_t out[16];
    for (unsigned i = 0; i < 16; ++i)
      v[i] = round == 0 ? edges[i] : bits((uint32_t)(next() >> 32));
    ondrix_it_quant(v, out);
    for (unsigned i = 0; i < 16; ++i)
      if (out[i] != quant_ref(v[i]))
        return fail("quant", round * 100 + (long)i);
  }

  static const int64_t wide[] = {INT32_MAX,   INT32_MIN,    2147483648, -2147483649,
                                 68719476736, -68719476736, INT64_MAX,  INT64_MIN};
  for (int round = 0; round < 1000; ++round) {
    int64_t v[8];
    int32_t out[8];
    for (unsigned i = 0; i < 8; ++i)
      v[i] = round == 0 ? wide[i] : (int64_t)next() >> (next() % 64);
    ondrix_it_narrow(v, out);
    for (unsigned i = 0; i < 8; ++i) {
      int32_t saturated = v[i] > INT32_MAX   ? INT32_MAX
                          : v[i] < INT32_MIN ? INT32_MIN
                                             : (int32_t)v[i];
      if (out[i] != (int32_t)((uint32_t)saturated - (uint32_t)v[i]))
        return fail("narrow", round * 10 + (long)i);
    }
  }

  for (int round = 0; round < 1000; ++round) {
    int32_t a[8];
    int64_t b[8];
    float out[8];
    for (unsigned i = 0; i < 8; ++i)
      a[i] = (int32_t)(next() >> 32), b[i] = (int64_t)next() >> (next() % 64);
    ondrix_it_widen(a, b, out);
    for (unsigned i = 0; i < 8; ++i)
      if (!same(out[i], (float)a[i] * 0.5f + (float)b[i]))
        return fail("widen", round * 10 + (long)i);
  }

  for (unsigned n = 1; n <= N; ++n) {
    int32_t x[N];
    int64_t y[4], expected[4] = {0, 0, 0, 0};
    for (unsigned i = 0; i < n; ++i)
      x[i] = (int32_t)(next() >> 32);
    for (unsigned i = 1; i < n; ++i)
      expected[(i - 1) % 4] += (int64_t)x[i] * 65536;
    memset(y, 0x5a, sizeof(y));
    ondrix_it_local(x, n, y, 4);
    if (memcmp(y, expected, sizeof(y)) != 0)
      return fail("local", n);
  }
  return 0;
}

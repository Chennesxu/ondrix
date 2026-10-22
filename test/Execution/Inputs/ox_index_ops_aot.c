#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#define N 40

static uint64_t state = 0x6a09e667f3bcc909u;

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

static float in[N], out[N];
static uint64_t length;
static int32_t mode;
static void traps(void) { ondrix_ix_traps(in, length, out, 8, mode); }
static void decimateShort(void) { ondrix_ix_decimate(in, 5, out, 3); }

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
  static int16_t q[N], qy[N + 1];
  for (unsigned i = 0; i < N; ++i) {
    in[i] = (float)((int32_t)(next() >> 40) - (1 << 23)) / 2048.0f;
    q[i] = (int16_t)(next() >> 48);
  }

  for (unsigned n = 0; 2 * n <= N; ++n) {
    out[n] = 77.0f;
    ondrix_ix_decimate(in, 2 * n, out, n);
    for (unsigned i = 0; i < n; ++i)
      if (!same(out[i], in[2 * i] + in[2 * i + 1]))
        return fail("decimate", i);
    if (out[n] != 77.0f)
      return fail("decimate guard", n);
  }
  if (!dies(decimateShort, "index out of bounds"))
    return fail("decimate trap", 0);

  // (i - 1) / 2 truncates toward zero: y[0] reads x[0], where flooring would read x[-1].
  for (unsigned n = 1; n <= N; ++n) {
    qy[n] = 77;
    ondrix_ix_hold(q, (n + 1) / 2, qy, n);
    for (unsigned i = 0; i < n; ++i)
      if (qy[i] != q[((int)i - 1) / 2])
        return fail("hold", i);
    if (qy[n] != 77)
      return fail("hold guard", n);
  }

  // Each mode has a length whose index product or quotient leaves the 64-bit index
  // range (or divides by zero), and a neighbouring length that stays inside it.
  static const struct {
    int32_t mode;
    uint64_t fits, traps;
    unsigned slot;
    float value;
    const char *message;
  } cases[] = {
      {0, 7, 8, 6, 1.0f, "index arithmetic overflow"},
      {1, 0, 1, 0, 2.0f, "index arithmetic overflow"},
      {2, 0, 1, 0, 3.0f, "integer division by zero or overflow"},
      {3, 1, 0, 7, 4.0f, "integer division by zero or overflow"},
  };
  for (unsigned i = 0; i < 4; ++i) {
    mode = cases[i].mode;
    memset(out, 0, sizeof out);
    ondrix_ix_traps(in, cases[i].fits, out, 8, mode);
    if (!same(out[cases[i].slot], cases[i].value))
      return fail("traps control", i);
    length = cases[i].traps;
    if (!dies(traps, cases[i].message))
      return fail("traps", i);
  }
  return 0;
}

#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

enum { kTaps = 16 };

static int16_t edge[kTaps] = {32767, 32767, 1}; // sums to 65535, the largest admitted
static int16_t past[kTaps] = {32767, 32767, 2}; // sums to 65536, one unit past
static int16_t signal_[kTaps];

// The declared default contract: exact products, ties-positive export to Q15.
static int16_t reference(const int16_t *x, const int16_t *h) {
  int64_t sum = 0;
  for (int i = 0; i < kTaps; ++i)
    sum += (int64_t)x[i] * h[i];
  int64_t rounded = (sum + (1 << 14)) >> 15;
  return (int16_t)(rounded > 32767 ? 32767 : rounded < -32768 ? -32768 : rounded);
}

// Runs the kernel on `past` in a child; true when it aborted after printing
// the checked-entry message.
static int refusesPast(void) {
  int pipefd[2];
  if (pipe(pipefd) != 0)
    return 0;
  pid_t child = fork();
  if (child == 0) {
    dup2(pipefd[1], STDOUT_FILENO);
    close(pipefd[0]);
    setvbuf(stdout, NULL, _IONBF, 0);
    (void)ondrix_q15_dot_gain_bound(signal_, past);
    _exit(0);
  }
  close(pipefd[1]);
  char output[256] = {0};
  size_t count = 0;
  for (ssize_t got = 1; got > 0 && count < sizeof(output) - 1; count += (size_t)got)
    got = read(pipefd[0], output + count, sizeof(output) - 1 - count);
  close(pipefd[0]);
  int status = 0;
  if (child < 0 || waitpid(child, &status, 0) != child)
    return 0;
  return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT &&
         strstr(output, "ondrix_q15_dot_gain_bound: coefficients exceed the declared gain_bound");
}

int main(void) {
  uint32_t state = 0x9E3779B9u;
  long failures = 0;
  for (int trial = 0; trial < 4000; ++trial) {
    for (int i = 0; i < kTaps; ++i) {
      state ^= state << 13, state ^= state >> 17, state ^= state << 5;
      signal_[i] = trial % 3 == 0 ? (int16_t)(i % 2 ? -32768 : 32767) : (int16_t)state;
    }
    if (ondrix_q15_dot_gain_bound(signal_, edge) != reference(signal_, edge))
      ++failures;
#if !CHECKED
    if (ondrix_q15_dot_gain_bound(signal_, past) != reference(signal_, past))
      ++failures;
#endif
  }
#if CHECKED
  if (!refusesPast()) {
    printf("the checked build ran a table past its declared gain\n");
    return 1;
  }
#endif
  printf("gain_bound %s: %ld failures\n", CHECKED ? "checked" : "trusted", failures);
  return failures != 0;
}

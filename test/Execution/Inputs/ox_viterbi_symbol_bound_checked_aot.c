#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

enum { kBits = 256, kBound = 127 };

static int16_t symbols[2 * kBits];
static uint8_t message[kBits / 8];

static int parity(unsigned value) {
  value ^= value >> 4;
  value ^= value >> 2;
  value ^= value >> 1;
  return value & 1;
}

// A noiseless (171,133) codeword at the declared bound decodes to its message.
static void encode(void) {
  uint32_t lcg = 0x2545F491u;
  unsigned state = 0;
  for (int n = 0; n < kBits; ++n) {
    lcg = lcg * 1664525u + 1013904223u;
    unsigned input = n < kBits - 6 ? (lcg >> 20) & 1 : 0;
    if (input)
      message[n / 8] |= (uint8_t)(0x80u >> (n % 8));
    unsigned reg = (input << 6) | state;
    symbols[2 * n] = parity(0171 & reg) ? -kBound : kBound;
    symbols[2 * n + 1] = parity(0133 & reg) ? -kBound : kBound;
    state = (state >> 1) | (input << 5);
  }
}

// Runs the decoder in a child; true when it aborted after printing the
// checked-entry message.
static int refuses(void) {
  int pipefd[2];
  if (pipe(pipefd) != 0)
    return 0;
  pid_t child = fork();
  if (child == 0) {
    dup2(pipefd[1], STDOUT_FILENO);
    close(pipefd[0]);
    setvbuf(stdout, NULL, _IONBF, 0);
    int8_t bits[kBits / 8];
    ondrix_viterbi_bounded(symbols, bits);
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
         strstr(output, "ondrix_viterbi_bounded: symbols exceed the declared symbol_bound");
}

int main(void) {
  encode();
  int8_t bits[kBits / 8];
  ondrix_viterbi_bounded(symbols, bits);
  if (memcmp(bits, message, sizeof message) != 0) {
    printf("FAIL: the frame at the bound did not decode\n");
    return 1;
  }
  symbols[77] = kBound + 1;
#if CHECKED
  if (!refuses()) {
    printf("FAIL: a symbol past the bound was not refused\n");
    return 1;
  }
#else
  ondrix_viterbi_bounded(symbols, bits);
#endif
  printf("viterbi symbol_bound: PASS\n");
  return 0;
}

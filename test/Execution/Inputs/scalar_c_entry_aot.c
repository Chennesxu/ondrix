#include <limits.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static int expectAbort(unsigned mode) {
  pid_t child = fork();
  if (child == 0) {
    float buffer[4] = {1, 2, 3, 4};
    if (mode == 0)
      ondrix_scalar_mixed(1, NULL, 2, buffer, 4, 3);
    else if (mode == 1)
      ondrix_scalar_mixed(1, buffer, 2, buffer + 1, 3, 3);
    else if (mode == 2)
      ondrix_scalar_mixed(1, buffer, 2, buffer, UINT64_MAX, 3);
    else
      ondrix_scalar_fill(7, NULL);
    _exit(0);
  }
  int status;
  return child > 0 && waitpid(child, &status, 0) == child && WIFSIGNALED(status) &&
         WTERMSIG(status) == SIGABRT;
}

int main(void) {
  if (ondrix_scalar_i32(INT32_MIN, 1) != INT32_MAX ||
      ondrix_scalar_i32(-1234567, 7654321) != -8888888 ||
      ondrix_scalar_i64(INT64_MIN, 1) != INT64_MAX ||
      ondrix_scalar_i64(INT64_C(0x123456789abcdef), INT64_C(0x123456780000000)) !=
          INT64_C(0x9abcdef)) {
    fprintf(stderr, "scalar signed integer ABI mismatch\n");
    return 1;
  }
  // Noncanonical external bytes must normalize by nonzero comparison, and
  // every result must contain exactly 0 or 1, including after LLVM O3.
  const uint8_t flags[] = {0, 1, 2, 127, 128, 254, 255};
  for (unsigned i = 0; i < sizeof(flags); ++i)
    for (unsigned j = 0; j < sizeof(flags); ++j)
      if (ondrix_scalar_bool(flags[i], flags[j]) != ((flags[i] != 0) != (flags[j] != 0))) {
        fprintf(stderr, "boolean byte normalization mismatch\n");
        return 1;
      }
  const uint32_t bits[] = {0, 0x80000000u, 1, 0x7f800000u, 0xff800000u, 0x7fc12345u, 0x3fc00000u};
  for (unsigned i = 0; i < sizeof(bits) / sizeof(bits[0]); ++i) {
    float input;
    memcpy(&input, &bits[i], sizeof(input));
    float output = ondrix_scalar_f32(input);
    uint32_t actual;
    memcpy(&actual, &output, sizeof(actual));
    if (actual != bits[i]) {
      fprintf(stderr, "scalar f32 ABI mismatch\n");
      return 1;
    }
  }
  float lhs[] = {1, 2, 3}, rhs[] = {4, 5, 6};
  if (ondrix_scalar_mixed(0.5f, lhs, 2, rhs, 3, -7) != 25.5f ||
      ondrix_scalar_mixed(0.5f, lhs, 0, rhs, 3, -7) != 0.5f ||
      ondrix_scalar_mixed(0.5f, NULL, 255, NULL, 0, -7) != -6.5f) {
    fprintf(stderr, "mixed scalar and dynamic buffer ABI mismatch\n");
    return 1;
  }
  int32_t output[2] = {0, 0};
  int32_t lengthsA[3] = {0, 0, 0};
  int16_t lengthsB[1] = {0};
  if (ondrix_scalar_lengths(lengthsA, 3, -5, lengthsB, 1, 128) != -3 ||
      ondrix_scalar_lengths(NULL, 0, -5, NULL, 0, 0) != -5) {
    fprintf(stderr, "independent dynamic lengths changed scalar ordering\n");
    return 1;
  }
  ondrix_scalar_fill(INT32_MIN, output);
  if (output[0] != INT32_MIN || output[1] != INT32_MIN) {
    fprintf(stderr, "scalar input with result buffer mismatch\n");
    return 1;
  }
  for (unsigned mode = 0; mode < 4; ++mode)
    if (!expectAbort(mode)) {
      fprintf(stderr, "mixed scalar/buffer guard %u did not abort\n", mode);
      return 1;
    }
  return 0;
}

#include "fixed_point_reference.h"

#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>

// The descriptor entry the plain-pointer entry is held against.
extern int16_t q15_dot(int16_t *lhs_allocated, int16_t *lhs_aligned, int64_t lhs_offset,
                       int64_t lhs_size, int64_t lhs_stride, int16_t *rhs_allocated,
                       int16_t *rhs_aligned, int64_t rhs_offset, int64_t rhs_size,
                       int64_t rhs_stride);

static const struct Policy q15_saturating = {
    .width = 16,
    .frac = 15,
    .accumulator_width = 40,
    .accumulator_frac = 30,
    .update_overflow = SATURATE,
    .state_rounding = NEAREST_EVEN,
    .state_overflow = SATURATE,
    .output_rounding = NEAREST_EVEN,
    .output_overflow = SATURATE,
};

static int16_t reference(const int16_t *lhs, const int16_t *rhs, int64_t count,
                         const struct Policy *policy) {
  int64_t accumulator = 0;
  for (int64_t i = 0; i < count; ++i)
    accumulator = update_reference(accumulator, lhs[i], rhs[i], policy);
  return (int16_t)export_reference(accumulator, policy->output_rounding, policy->output_overflow,
                                   policy);
}

// A refused length must abort before any element is read: the buffers are null.
static int refuses(uint64_t length) {
  pid_t child = fork();
  if (child == 0) {
    freopen("/dev/null", "w", stdout);
    ondrix_q15_dot(NULL, NULL, length);
    _exit(0);
  }
  int status = 0;
  if (child < 0 || waitpid(child, &status, 0) != child)
    return 0;
  return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
}

int main(void) {
  int16_t lhs[] = {-32768, 32767, 16384, -16384, 1, -1, 12345, -23456, 32767};
  int16_t rhs[] = {-32768, 32767, 16384, 16384, 16384, 16384, -22222, 11111, -32768};
  uint64_t lengths[] = {0, 1, 2, 3, 8, 9};
  for (unsigned i = 0; i < sizeof(lengths) / sizeof(lengths[0]); ++i) {
    int64_t length = (int64_t)lengths[i];
    int16_t entry = ondrix_q15_dot(lhs, rhs, lengths[i]);
    int16_t direct = q15_dot(lhs, lhs, 0, length, 1, rhs, rhs, 0, length, 1);
    int16_t expected = reference(lhs, rhs, length, &q15_saturating);
    if (entry != direct || entry != expected) {
      fprintf(stderr, "q15 dot length %lld: entry %d, descriptor %d, expected %d\n",
              (long long)length, entry, direct, expected);
      return 1;
    }
  }

  static const int16_t coefficients[5] = {16384, -8192, 4096, -8192, 16384};
  struct Policy wrapping = q15_saturating;
  wrapping.update_overflow = WRAP;
  int16_t windows[][5] = {
      {0, 0, 0, 0, 0},
      {-32768, 32767, 16384, -16384, 1},
      {32767, 32767, 32767, 32767, 32767},
      {-12345, 23456, -1, 1, 30000},
  };
  for (unsigned i = 0; i < sizeof(windows) / sizeof(windows[0]); ++i) {
    int16_t entry = ondrix_q15_fir_constexpr(windows[i]);
    int16_t expected = reference(windows[i], coefficients, 5, &wrapping);
    if (entry != expected) {
      fprintf(stderr, "constexpr q15 fir window %u: entry %d, expected %d\n", i, entry, expected);
      return 1;
    }
  }

  if (!refuses(UINT64_MAX) || !refuses((uint64_t)1 << 63)) {
    fprintf(stderr, "a length past the signed index range was not refused\n");
    return 1;
  }
  return 0;
}

/* Independent reference for the ondsp.round_quotient contract: round_div's
 * sequence over a runtime divisor, and the saturating non-positive policy
 * (the dividend's signed rail, zero for a zero dividend). */

#include <inttypes.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>

extern int16_t rq16_ratio_tp(int16_t, int16_t);
extern int16_t rq16_ratio_ne(int16_t, int16_t);
extern int16_t rq16_ratio_floor(int16_t, int16_t);
extern int16_t rq16_ratio_zero(int16_t, int16_t);
extern int16_t rq16_plain_tp(int16_t, int16_t);
extern int16_t rq16_plain_ne(int16_t, int16_t);
extern int16_t rq32_ratio_wrap(int32_t, int16_t);
extern int16_t rq16_ratio_trap(int16_t, int16_t);

enum Mode { kFloor, kZero, kTiesPositive, kNearestEven };
enum Overflow { kSaturate, kWrap };

static int64_t reference(int64_t x, int64_t d, unsigned preShift, enum Mode mode,
                         enum Overflow overflow, unsigned resultWidth) {
  __int128 low = -((__int128)1 << (resultWidth - 1));
  __int128 high = ((__int128)1 << (resultWidth - 1)) - 1;
  if (d <= 0)
    return x == 0 ? 0 : (int64_t)(x < 0 ? low : high);
  __int128 scaled = (__int128)x << preShift;
  __int128 quotient = scaled / d, remainder = scaled % d;
  if (remainder < 0) {
    --quotient;
    remainder += d;
  }
  switch (mode) {
  case kFloor:
    break;
  case kZero:
    quotient += scaled < 0 && remainder != 0;
    break;
  case kTiesPositive:
    quotient += remainder >= d - remainder;
    break;
  case kNearestEven:
    quotient += remainder > d - remainder || (remainder == d - remainder && (quotient & 1));
    break;
  }
  if (overflow == kSaturate)
    return (int64_t)(quotient < low ? low : (quotient > high ? high : quotient));
  uint64_t mask = (UINT64_C(1) << resultWidth) - 1, bits = (uint64_t)quotient & mask;
  if (bits & (UINT64_C(1) << (resultWidth - 1)))
    bits |= ~mask;
  return (int64_t)bits;
}

static int failed;

static void expectEqual(const char *label, int64_t x, int64_t d, int64_t actual, int64_t expected) {
  if (actual != expected && !failed) {
    fprintf(stderr, "%s(%" PRId64 ", %" PRId64 "): got %" PRId64 ", expected %" PRId64 "\n", label,
            x, d, actual, expected);
    failed = 1;
  }
}

struct Config {
  const char *label;
  int16_t (*kernel)(int16_t, int16_t);
  unsigned preShift;
  enum Mode mode;
};

// Runs the trapping kernel on a divisor that is not positive in a child and
// reports whether the child stopped with SIGABRT.
static int traps(int16_t x, int16_t d) {
  pid_t child = fork();
  if (child == 0) {
    freopen("/dev/null", "w", stdout);
    freopen("/dev/null", "w", stderr);
    volatile int16_t sink = rq16_ratio_trap(x, d);
    (void)sink;
    _exit(0);
  }
  int status = 0;
  if (child < 0 || waitpid(child, &status, 0) != child)
    return 0;
  return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
}

int main(void) {
  static const struct Config configs[] = {
      {"rq16_ratio_tp", rq16_ratio_tp, 15, kTiesPositive},
      {"rq16_ratio_ne", rq16_ratio_ne, 15, kNearestEven},
      {"rq16_ratio_floor", rq16_ratio_floor, 15, kFloor},
      {"rq16_ratio_zero", rq16_ratio_zero, 15, kZero},
      {"rq16_plain_tp", rq16_plain_tp, 0, kTiesPositive},
      {"rq16_plain_ne", rq16_plain_ne, 0, kNearestEven},
  };
  static const int16_t divisors[] = {1, 2, 3, 6, 7, 32767, 0, -1, -3, -32768};
  static const int16_t dividends[] = {0, 1, -1, 3, -3, 9, -9, 12345, 32767, -32768};
  int ratioTie = 0, plainTie = 0, nonpositiveSeen = 0, floorDiffers = 0;
  for (unsigned c = 0; c < sizeof(configs) / sizeof(configs[0]); ++c) {
    const struct Config *config = &configs[c];
    for (unsigned k = 0; k < sizeof(divisors) / sizeof(divisors[0]); ++k)
      for (int32_t raw = INT16_MIN; raw <= INT16_MAX; ++raw) {
        int16_t x = (int16_t)raw, d = divisors[k];
        expectEqual(config->label, x, d, config->kernel(x, d),
                    reference(x, d, config->preShift, config->mode, kSaturate, 16));
        nonpositiveSeen |= d <= 0 && x != 0;
      }
    for (unsigned k = 0; k < sizeof(dividends) / sizeof(dividends[0]); ++k)
      for (int32_t raw = INT16_MIN; raw <= INT16_MAX; ++raw) {
        int16_t x = dividends[k], d = (int16_t)raw;
        expectEqual(config->label, x, d, config->kernel(x, d),
                    reference(x, d, config->preShift, config->mode, kSaturate, 16));
      }
  }
  // Positive controls. The unscaled even divisor reaches the tie, so its two
  // nearest kernels differ; the ratio's tie is unreachable, so its two
  // nearest kernels agree everywhere while floor still differs from them.
  for (int32_t raw = INT16_MIN; raw <= INT16_MAX; ++raw) {
    int16_t x = (int16_t)raw;
    plainTie |= rq16_plain_tp(x, 6) != rq16_plain_ne(x, 6);
    for (unsigned k = 0; k < sizeof(divisors) / sizeof(divisors[0]); ++k) {
      ratioTie |= rq16_ratio_tp(x, divisors[k]) != rq16_ratio_ne(x, divisors[k]);
      floorDiffers |= rq16_ratio_tp(x, divisors[k]) != rq16_ratio_floor(x, divisors[k]);
    }
  }
  if (!plainTie || ratioTie || !floorDiffers || !nonpositiveSeen) {
    fprintf(stderr,
            "positive controls: plain tie %d, ratio tie %d, floor differs %d, "
            "non-positive seen %d\n",
            plainTie, ratioTie, floorDiffers, nonpositiveSeen);
    return 1;
  }
  // Wrap narrowing from a wider dividend: directed quotients past both rails.
  static const int32_t wide[] = {INT32_MAX, INT32_MIN, 65536, -65537, 40000, -40000, 3, -3, 0};
  for (unsigned i = 0; i < sizeof(wide) / sizeof(wide[0]); ++i)
    for (unsigned k = 0; k < sizeof(divisors) / sizeof(divisors[0]); ++k)
      expectEqual("rq32_ratio_wrap", wide[i], divisors[k], rq32_ratio_wrap(wide[i], divisors[k]),
                  reference(wide[i], divisors[k], 15, kNearestEven, kWrap, 16));
  // The trapping policy agrees with the saturating one wherever the divisor
  // is positive, and stops the process wherever it is not.
  for (int32_t raw = 1; raw <= INT16_MAX; raw += 97)
    for (unsigned k = 0; k < sizeof(dividends) / sizeof(dividends[0]); ++k)
      expectEqual("rq16_ratio_trap", dividends[k], raw, rq16_ratio_trap(dividends[k], (int16_t)raw),
                  rq16_ratio_tp(dividends[k], (int16_t)raw));
  if (!traps(12345, 0) || !traps(-3, -7) || traps(12345, 1)) {
    fprintf(stderr, "the trapping policy did not stop exactly on the non-positive divisors\n");
    return 1;
  }
  return failed;
}

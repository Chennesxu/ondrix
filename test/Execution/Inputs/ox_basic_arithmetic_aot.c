#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

// Use unsigned arithmetic for every potentially overflowing operation and
// memcpy for signed bit interpretation; the reference never executes C UB.
static int32_t signed32(uint32_t bits) {
  int32_t value;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

static int64_t signed64(uint64_t bits) {
  int64_t value;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

static int32_t numerator32(int32_t x, int32_t y, int32_t z) {
  uint32_t t = (uint32_t)x * (uint32_t)y + (uint32_t)z - (uint32_t)x;
  return signed32(0u - t);
}

static int64_t numerator64(int64_t x, int64_t y, int64_t z) {
  uint64_t t = (uint64_t)x * (uint64_t)y + (uint64_t)z - (uint64_t)x;
  return signed64(UINT64_C(0) - t);
}

static uint64_t next(uint64_t *state) {
  *state ^= *state << 13;
  *state ^= *state >> 7;
  *state ^= *state << 17;
  return *state;
}

static int integers(void) {
  const int64_t rails[] = {0, 1, -1, 2, -2, INT32_MIN, INT32_MAX, INT64_MIN, INT64_MAX};
  uint64_t state = UINT64_C(0x8f2736b1cde40597);
  for (unsigned i = 0; i < 400; ++i) {
    int64_t x = i < 81 ? rails[i / 9] : signed64(next(&state));
    int64_t y = i < 81 ? rails[i % 9] : signed64(next(&state));
    int64_t z = i < 81 ? rails[(i + 3) % 9] : signed64(next(&state));
    int64_t d = rails[(i + 1) % 9];
    if (d == 0)
      d = 7;
    int64_t n = numerator64(x, y, z);
    if (!(n == INT64_MIN && d == -1) && ondrix_basic_i64(x, y, z, d) != n / d) {
      fprintf(stderr, "i64 wrap/quotient mismatch at case %u\n", i);
      return 1;
    }
    int32_t a = signed32((uint32_t)x), b = signed32((uint32_t)y);
    int32_t c = signed32((uint32_t)z), divisor = signed32((uint32_t)d);
    if (divisor == 0)
      divisor = -7;
    int32_t numerator = numerator32(a, b, c);
    if (!(numerator == INT32_MIN && divisor == -1) &&
        ondrix_basic_i32(a, b, c, divisor) != numerator / divisor) {
      fprintf(stderr, "i32 wrap/quotient mismatch at case %u\n", i);
      return 1;
    }
  }
  // Negating MIN wraps back to MIN, while negative quotients truncate to zero.
  return ondrix_basic_i32(0, 0, INT32_MIN, 1) != INT32_MIN ||
         ondrix_basic_i64(0, 0, INT64_MIN, 1) != INT64_MIN || ondrix_basic_i32(0, 0, 7, 3) != -2 ||
         ondrix_basic_i64(0, 0, -7, 3) != 2;
}

static int booleans(void) {
  const uint8_t flags[] = {0, 1, 2, 127, 128, 254, 255};
  for (unsigned i = 0; i < sizeof(flags); ++i)
    for (unsigned j = 0; j < sizeof(flags); ++j) {
      uint8_t a = flags[i], b = flags[j];
      if (ondrix_basic_bool(0, 0, a, b, 2) != ((a != 0) == (b != 0)) ||
          ondrix_basic_bool(0, 0, a, b, 3) != ((a != 0) != (b != 0)) ||
          ondrix_basic_bool(0, 0, a, b, 4) != (a == 0) || ondrix_basic_bool(0, 0, a, b, 11) != 1)
        return 1;
    }
  const int64_t values[] = {INT64_MIN, -1, 0, 1, INT64_MAX};
  for (unsigned i = 0; i < 5; ++i)
    for (unsigned j = 0; j < 5; ++j) {
      int64_t x = values[i], y = values[j];
      const uint8_t expected[] = {x == y, x != y, x<y, x <= y, x> y, x >= y};
      for (unsigned mode = 0; mode < 6; ++mode)
        if (ondrix_basic_bool(x, y, 0, 0, (int32_t)mode + 5) != expected[mode])
          return 1;
    }
  // Both invalid division classes must remain unevaluated on skipped RHSs.
  return ondrix_basic_bool(1, 0, 0, 0, 0) != 0 || ondrix_basic_bool(INT64_MIN, -1, 0, 0, 0) != 0 ||
         ondrix_basic_bool(1, 0, 255, 0, 1) != 1 ||
         ondrix_basic_bool(INT64_MIN, -1, 2, 0, 1) != 1 ||
         ondrix_basic_bool(7, 3, 128, 0, 0) != 1 || ondrix_basic_bool(-7, 3, 0, 0, 1) != 0 ||
         ondrix_basic_eager(7, 3, 255) != 1 || ondrix_basic_eager(7, 3, 0) != 0;
}

static float fromBits(uint32_t bits) {
  float result;
  memcpy(&result, &bits, sizeof(result));
  return result;
}

static uint32_t toBits(float value) {
  uint32_t result;
  memcpy(&result, &value, sizeof(result));
  return result;
}

static int sameFloat(float actual, float expected) {
  return isnan(expected) ? isnan(actual) : toBits(actual) == toBits(expected);
}

// Volatile binary32 stages prevent excess precision or contraction in the C
// authority. FP exceptions and NaN payloads are outside the source contract.
static float referenceFloat(float x, float y, float z, float w, float d) {
  volatile float product = x * y;
  volatile float total = product + z;
  volatile float difference = total - w;
  volatile float negative = -difference;
  volatile float result = negative / d;
  return result;
}

static int floats(void) {
  const uint32_t corpus[] = {0,           0x80000000u, 1,           0x80000001u, 0x007fffffu,
                             0x00800000u, 0x3f800000u, 0xbf800000u, 0x3f800001u, 0x3f7ffffeu,
                             0x7f7fffffu, 0x7f800000u, 0xff800000u, 0x7fc12345u};
  unsigned count = sizeof(corpus) / sizeof(corpus[0]);
  for (unsigned i = 0; i < count; ++i) {
    float x = fromBits(corpus[i]);
    if (!sameFloat(ondrix_basic_f32_neg(x), -x))
      return 1;
    for (unsigned j = 0; j < count; ++j) {
      float y = fromBits(corpus[j]);
      float z = fromBits(corpus[(i + 3) % count]);
      float w = fromBits(corpus[(j + 7) % count]);
      float actual = ondrix_basic_f32(x, y, z, w, y);
      if (!sameFloat(actual, referenceFloat(x, y, z, w, y))) {
        fprintf(stderr, "f32 arithmetic mismatch at (%u,%u)\n", i, j);
        return 1;
      }
      const uint8_t expected[] = {x == y, x != y, x<y, x <= y, x> y, x >= y};
      for (unsigned mode = 0; mode < 6; ++mode)
        if (ondrix_basic_f32_compare(x, y, (int32_t)mode) != expected[mode])
          return 1;
    }
  }
  // This product rounds to 1 before subtraction; an FMA yields a nonzero
  // residual. Large cancellation also distinguishes source association.
  if (toBits(ondrix_basic_f32(fromBits(0x3f800001u), fromBits(0x3f7ffffeu), -1, 0, 1)) !=
          0x80000000u ||
      !sameFloat(ondrix_basic_f32(0x1p24f, 1, 1, 0x1p24f, 1), -0.0f) ||
      !sameFloat(ondrix_basic_f32(fromBits(1), 1, 0, 0, 1), fromBits(0x80000001u)) ||
      !sameFloat(ondrix_basic_f32(1, 1, 0, 0, 0), -INFINITY) ||
      !isnan(ondrix_basic_f32(0, 1, 0, 0, 0)))
    return 1;
  for (unsigned base = 0; base < count; ++base) {
    float x[4], y[4], output[4];
    for (unsigned lane = 0; lane < 4; ++lane) {
      x[lane] = fromBits(corpus[(base + lane) % count]);
      y[lane] = fromBits(corpus[(base * 3 + lane + 1) % count]);
    }
    ondrix_basic_f32_tensor(x, y, output);
    for (unsigned lane = 0; lane < 4; ++lane)
      if (!sameFloat(output[lane], referenceFloat(x[lane], 0.5f, y[lane], 1, 2)))
        return 1;
  }
  return 0;
}

static int fixedNegation(void) {
  const int16_t q15[] = {INT16_MIN, INT16_MAX, -1, 0};
  const int32_t q31[] = {INT32_MIN, INT32_MAX, -1, 0};
  int16_t result15[4];
  int32_t result31[4];
  ondrix_basic_q15_neg(q15, result15);
  ondrix_basic_q31_neg(q31, result31);
  for (unsigned i = 0; i < 4; ++i) {
    int16_t expected15 = q15[i] == INT16_MIN ? INT16_MAX : -q15[i];
    int32_t expected31 = q31[i] == INT32_MIN ? INT32_MAX : -q31[i];
    if (result15[i] != expected15 || result31[i] != expected31)
      return 1;
  }
  return 0;
}

static int traps(unsigned mode) {
  pid_t child = fork();
  if (child == 0) {
    switch (mode) {
    case 0:
      ondrix_basic_i32(1, 2, 3, 0);
      break;
    case 1:
      ondrix_basic_i32(0, 0, INT32_MIN, -1);
      break;
    case 2:
      ondrix_basic_i64(1, 2, 3, 0);
      break;
    case 3:
      ondrix_basic_i64(0, 0, INT64_MIN, -1);
      break;
    case 4:
      ondrix_basic_bool(1, 0, 1, 0, 0);
      break;
    case 5:
      ondrix_basic_bool(INT64_MIN, -1, 1, 0, 0);
      break;
    case 6:
      ondrix_basic_bool(1, 0, 0, 0, 1);
      break;
    case 7:
      ondrix_basic_bool(INT64_MIN, -1, 0, 0, 1);
      break;
    case 8:
      ondrix_basic_eager(1, 0, 0);
      break;
    case 9:
      ondrix_basic_eager(INT64_MIN, -1, 0);
      break;
    }
    _exit(0);
  }
  int status;
  // SIGFPE would permit an unchecked native division; require an explicit
  // compiler assertion (SIGABRT) or trap instruction (SIGILL) instead.
  return child > 0 && waitpid(child, &status, 0) == child && WIFSIGNALED(status) &&
         (WTERMSIG(status) == SIGABRT || WTERMSIG(status) == SIGILL);
}

int main(void) {
  if (integers() || booleans() || floats() || fixedNegation()) {
    fprintf(stderr, "source arithmetic differential check failed\n");
    return 1;
  }
  for (unsigned mode = 0; mode < 10; ++mode)
    if (!traps(mode)) {
      fprintf(stderr, "checked division mode %u did not trap\n", mode);
      return 1;
    }
  return 0;
}

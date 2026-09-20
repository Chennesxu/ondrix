#include <stdint.h>
#include <stdio.h>

extern int64_t ortumcore_repeat_mac(int16_t lhs, int16_t rhs, int64_t count);
extern int64_t ortumcore_repeat_mac_sub(int16_t lhs, int16_t rhs, int64_t count);
extern int32_t ortumcore_repeat_mac_out_q15(int16_t lhs, int16_t rhs, int64_t count);
extern int32_t ortumcore_repeat_mac_out_raw(int16_t lhs, int16_t rhs, int64_t count);
extern int32_t ortumcore_bitrev_add_walk(int32_t start, int32_t step, int64_t count);
extern int32_t ortumcore_bitrev_sub_walk(int32_t start, int32_t step, int64_t count);
extern int32_t ortumcore_dmac_walk(int16_t lhs0, int16_t rhs0, int16_t lhs1, int16_t rhs1,
                                   int64_t count, int32_t lane);

static int64_t update_reference(int64_t accumulator, int16_t lhs, int16_t rhs, int subtract) {
  const __int128 minimum = -((__int128)1 << 39);
  const __int128 maximum = ((__int128)1 << 39) - 1;
  __int128 product = (__int128)lhs * (__int128)rhs;
  __int128 updated = subtract ? (__int128)accumulator - product : (__int128)accumulator + product;
  if (updated < minimum)
    return (int64_t)minimum;
  if (updated > maximum)
    return (int64_t)maximum;
  return (int64_t)updated;
}

static int64_t repeat_reference(int16_t lhs, int16_t rhs, int64_t count, int subtract) {
  int64_t accumulator = 0;
  for (int64_t i = 0; i < count; ++i)
    accumulator = update_reference(accumulator, lhs, rhs, subtract);
  return accumulator;
}

// Independent readout formulation: floor division (not >>), then the i32
// clamp, mirroring the manual equation rather than the compiler's lowering.
// Independent bit-order formulation: digit walk, not the butterfly swaps
// the emulation lowering uses.
static uint32_t reverse_reference(uint32_t value) {
  uint32_t reversed = 0;
  for (int bit = 0; bit < 32; ++bit)
    reversed |= ((value >> bit) & 1u) << (31 - bit);
  return reversed;
}

static int32_t bitrev_walk_reference(int32_t start, int32_t step, int64_t count, int subtract) {
  uint32_t address = (uint32_t)start;
  for (int64_t i = 0; i < count; ++i) {
    uint32_t raw = subtract ? reverse_reference(address) - reverse_reference((uint32_t)step)
                            : reverse_reference(address) + reverse_reference((uint32_t)step);
    address = reverse_reference(raw);
  }
  return (int32_t)address;
}

static int32_t readout_reference(int64_t accumulator, unsigned shift) {
  __int128 divisor = (__int128)1 << shift;
  __int128 quotient = (__int128)accumulator / divisor;
  if ((__int128)accumulator % divisor != 0 && accumulator < 0)
    --quotient;
  if (quotient < INT32_MIN)
    return INT32_MIN;
  if (quotient > INT32_MAX)
    return INT32_MAX;
  return (int32_t)quotient;
}

static int check(const char *name, int16_t lhs, int16_t rhs, int64_t count, int subtract) {
  int64_t expected = repeat_reference(lhs, rhs, count, subtract);
  int64_t actual =
      subtract ? ortumcore_repeat_mac_sub(lhs, rhs, count) : ortumcore_repeat_mac(lhs, rhs, count);
  if (actual == expected)
    return 0;
  fprintf(stderr, "%s(%lld): expected %lld, got %lld\n", name, (long long)count,
          (long long)expected, (long long)actual);
  return 1;
}

int32_t ortumcore_cx_power(int32_t value, int32_t shift, int32_t ties);

int main(void) {
  static const int64_t counts[] = {0, 1, 2, 511, 512, 513};
  int failed = 0;
  for (unsigned i = 0; i < sizeof(counts) / sizeof(counts[0]); ++i) {
    failed |= check("target MAC", INT16_MIN, INT16_MIN, counts[i], 0);
    failed |= check("target MAC-sub", INT16_MIN, INT16_MIN, counts[i], 1);
  }
  failed |= check("mixed-sign target MAC", INT16_MIN, INT16_MAX, 19, 0);
  failed |= check("mixed-sign target MAC-sub", INT16_MIN, INT16_MAX, 19, 1);

  // Readout witnesses. shift 15 on a negative accumulator with a nonzero
  // remainder separates floor from truncation (-1 stays -1, truncation gives
  // 0); shift 0 on the saturated accumulator separates sat32 from plain
  // truncation (2^39-1 truncates to -1 but must clamp to 2^31-1).
  struct ReadoutCase {
    const char *name;
    int16_t lhs;
    int16_t rhs;
    int64_t count;
    unsigned shift;
  };
  static const struct ReadoutCase readout_cases[] = {
      {"readout floor negative", -1, 1, 1, 15},
      {"readout positive", 23170, 23170, 3, 15},
      {"readout negative walk", INT16_MIN, INT16_MAX, 7, 15},
      {"readout saturated high", INT16_MIN, INT16_MIN, 512, 0},
      {"readout saturated low", INT16_MIN, INT16_MAX, 512, 0},
      {"readout zero", 0, 0, 4, 0},
  };
  for (unsigned i = 0; i < sizeof(readout_cases) / sizeof(readout_cases[0]); ++i) {
    const struct ReadoutCase *c = &readout_cases[i];
    int64_t accumulator = repeat_reference(c->lhs, c->rhs, c->count, 0);
    int32_t expected = readout_reference(accumulator, c->shift);
    int32_t actual = c->shift == 15 ? ortumcore_repeat_mac_out_q15(c->lhs, c->rhs, c->count)
                                    : ortumcore_repeat_mac_out_raw(c->lhs, c->rhs, c->count);
    if (actual != expected) {
      fprintf(stderr, "%s: expected %d, got %d\n", c->name, expected, actual);
      failed = 1;
    }
  }
  // The floor witness must discriminate: truncation of the -1 accumulator
  // at shift 15 yields 0, floor yields -1.
  if (readout_reference(-1, 15) != -1) {
    fprintf(stderr, "readout reference lost the floor semantics\n");
    failed = 1;
  }
  struct BitrevCase {
    const char *name;
    int32_t start;
    int32_t step;
    int64_t count;
    int subtract;
  } bitrev_cases[] = {
      {"reversed counter walk", 0, INT32_MIN, 63, 0},
      {"reversed counter wrap down", 0, INT32_MIN, 1, 1},
      {"reversed long walk", 0, INT32_MIN, 1024, 0},
      {"reversed mixed operands", 0x12345678, 0x00010000, 17, 0},
      {"reversed mixed subtract", 0x12345678, (int32_t)0xAAAAAAAA, 9, 1},
  };
  for (unsigned i = 0; i < sizeof(bitrev_cases) / sizeof(bitrev_cases[0]); ++i) {
    const struct BitrevCase *c = &bitrev_cases[i];
    int32_t expected = bitrev_walk_reference(c->start, c->step, c->count, c->subtract);
    int32_t actual = c->subtract ? ortumcore_bitrev_sub_walk(c->start, c->step, c->count)
                                 : ortumcore_bitrev_add_walk(c->start, c->step, c->count);
    if (actual != expected) {
      fprintf(stderr, "%s: expected %d, got %d\n", c->name, expected, actual);
      failed = 1;
    }
  }
  // The reversed-counter walk must discriminate ordinary addition: after one
  // +rev32(1) step the address is 1 << 31, not 1.
  if (bitrev_walk_reference(0, INT32_MIN, 1, 0) != INT32_MIN) {
    fprintf(stderr, "bitrev reference lost the reversed-domain semantics\n");
    failed = 1;
  }

  // Dual-lane walks: each lane is exactly the scalar repeat reference on its
  // own pair, so lane independence is the whole claim being checked.
  struct DmacCase {
    const char *name;
    int16_t lhs0;
    int16_t rhs0;
    int16_t lhs1;
    int16_t rhs1;
    int64_t count;
  } dmac_cases[] = {
      {"lanes with opposite signs", INT16_MIN, INT16_MIN, INT16_MIN, INT16_MAX, 19},
      {"lane0 at the high rail", INT16_MIN, INT16_MIN, 1, -1, 513},
      {"lane1 at the low rail", 12345, -23456, INT16_MIN, INT16_MAX, 512},
      {"zero against walk", 0, 0, 23170, 23170, 40},
      {"single step", 32767, 32767, -32768, 32767, 1},
  };
  for (unsigned i = 0; i < sizeof(dmac_cases) / sizeof(dmac_cases[0]); ++i) {
    const struct DmacCase *c = &dmac_cases[i];
    for (int lane = 0; lane < 2; ++lane) {
      int64_t accumulator = lane == 0 ? repeat_reference(c->lhs0, c->rhs0, c->count, 0)
                                      : repeat_reference(c->lhs1, c->rhs1, c->count, 0);
      int32_t expected = readout_reference(accumulator, 15);
      int32_t actual = ortumcore_dmac_walk(c->lhs0, c->rhs0, c->lhs1, c->rhs1, c->count, lane);
      if (actual != expected) {
        fprintf(stderr, "%s lane %d: expected %d, got %d\n", c->name, lane, expected, actual);
        failed = 1;
      }
    }
  }
  static const struct PowerCase {
    const char *name;
    uint32_t packed;
  } power_cases[] = {
      {"both halves at the component minimum", 0x80008000u},
      {"a tie at the declared shift", 0xEA80F700u},
      {"another tie", 0x3C807200u},
      {"both halves at the maximum", 0x7FFF7FFFu},
      {"zero", 0x00000000u},
      {"asymmetric halves", 0x0001FFFFu},
  };
  int saturating = 0;
  int separated = 0;
  for (unsigned i = 0; i < sizeof(power_cases) / sizeof(power_cases[0]); ++i) {
    const struct PowerCase *c = &power_cases[i];
    int64_t real = (int16_t)(uint16_t)c->packed;
    int64_t imaginary = (int16_t)(uint16_t)(c->packed >> 16);
    int64_t sum = real * real + imaginary * imaginary;
    for (int shift = 0; shift <= 15; shift += 15) {
      for (int ties = 0; ties < 2; ++ties) {
        int64_t scaled =
            shift == 0 ? sum : (ties ? (sum + (INT64_C(1) << (shift - 1))) >> shift : sum >> shift);
        int32_t expected = scaled > INT32_MAX   ? INT32_MAX
                           : scaled < INT32_MIN ? INT32_MIN
                                                : (int32_t)scaled;
        if (scaled > INT32_MAX)
          saturating = 1;
        int32_t actual = ortumcore_cx_power((int32_t)c->packed, shift, ties);
        if (actual != expected) {
          fprintf(stderr, "power %s shift %d ties %d: expected %d, got %d\n", c->name, shift, ties,
                  expected, actual);
          failed = 1;
        }
      }
    }
    int64_t floored = sum >> 15;
    int64_t rounded = (sum + (INT64_C(1) << 14)) >> 15;
    if (floored != rounded)
      separated = 1;
  }
  if (!saturating) {
    fprintf(stderr, "power corpus never reaches the 32-bit saturation\n");
    failed = 1;
  }
  if (!separated) {
    fprintf(stderr, "power corpus cannot separate the two tie rules\n");
    failed = 1;
  }

  // The opposite-signs case must discriminate lane crosstalk: the two lane
  // readouts differ, so a swapped or shared accumulator cannot pass.
  if (readout_reference(repeat_reference(INT16_MIN, INT16_MIN, 19, 0), 15) ==
      readout_reference(repeat_reference(INT16_MIN, INT16_MAX, 19, 0), 15)) {
    fprintf(stderr, "dmac corpus cannot discriminate the lanes\n");
    failed = 1;
  }
  return failed;
}

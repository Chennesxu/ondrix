#include <stdint.h>
#include <stdio.h>

extern int64_t ortumcore_repeat_mac(int16_t lhs, int16_t rhs, int64_t count);
extern int64_t ortumcore_repeat_mac_sub(int16_t lhs, int16_t rhs, int64_t count);
extern int32_t ortumcore_repeat_mac_out_q15(int16_t lhs, int16_t rhs, int64_t count);
extern int32_t ortumcore_repeat_mac_out_raw(int16_t lhs, int16_t rhs, int64_t count);
extern int32_t ortumcore_bitrev_add_walk(int32_t start, int32_t step, int64_t count);
extern int32_t ortumcore_bitrev_sub_walk(int32_t start, int32_t step, int64_t count);

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
  return failed;
}

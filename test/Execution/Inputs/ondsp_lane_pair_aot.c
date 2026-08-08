#include "fixed_point_reference.h"

#include <stdio.h>

extern int32_t ondsp_pair_walk_floor(int16_t value0, int16_t value1, int16_t coefficient,
                                     int64_t count, int32_t lane);
extern int32_t ondsp_pair_walk_ntp(int16_t value0, int16_t value1, int16_t coefficient,
                                   int64_t count, int32_t lane);

static const struct Policy kPolicy = {
    32, 15, 40, 30, SATURATE, TOWARD_NEGATIVE, SATURATE, TOWARD_NEGATIVE, SATURATE};

static int64_t lane_reference(int16_t value, int16_t coefficient, int64_t count) {
  int64_t accumulator = 0;
  for (int64_t i = 0; i < count; ++i)
    accumulator = update_reference(accumulator, value, coefficient, &kPolicy);
  return accumulator;
}

int main(void) {
  // The shared coefficient is the declared broadcast; lanes differ only in
  // their value stream, exactly as the lanes = 2 contract states.
  struct PairCase {
    const char *name;
    int16_t value0;
    int16_t value1;
    int16_t coefficient;
    int64_t count;
  } cases[] = {
      {"opposite walks", -32768, 32767, -32768, 19},
      {"lane0 at the high rail", -32768, 1, -32768, 513},
      {"lane1 at the low rail", 23170, -32768, 32767, 512},
      {"negative and positive ties", 1, -1, -16384, 3},
      {"zero against single step", 0, 32767, 32767, 1},
  };
  int failed = 0;
  for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    const struct PairCase *c = &cases[i];
    for (int32_t lane = 0; lane < 2; ++lane) {
      int16_t value = lane == 0 ? c->value0 : c->value1;
      int64_t accumulator = lane_reference(value, c->coefficient, c->count);
      int32_t expected_floor =
          (int32_t)export_reference(accumulator, TOWARD_NEGATIVE, SATURATE, &kPolicy);
      int32_t expected_ntp =
          (int32_t)export_reference(accumulator, NEAREST_TIES_POSITIVE, SATURATE, &kPolicy);
      int32_t actual_floor =
          ondsp_pair_walk_floor(c->value0, c->value1, c->coefficient, c->count, lane);
      int32_t actual_ntp =
          ondsp_pair_walk_ntp(c->value0, c->value1, c->coefficient, c->count, lane);
      if (actual_floor != expected_floor) {
        fprintf(stderr, "%s lane %d floor: expected %d, got %d\n", c->name, lane, expected_floor,
                actual_floor);
        failed = 1;
      }
      if (actual_ntp != expected_ntp) {
        fprintf(stderr, "%s lane %d ntp: expected %d, got %d\n", c->name, lane, expected_ntp,
                actual_ntp);
        failed = 1;
      }
    }
  }

  // The corpus must discriminate what the pair path claims: independent lanes
  // (opposite walks differ), the rail carry (ties-positive exceeds floor at
  // the saturated rail), and the negative tie (ties-positive differs from
  // both floor and nearest-even).
  int64_t opposite0 = lane_reference(-32768, -32768, 19);
  int64_t opposite1 = lane_reference(32767, -32768, 19);
  if (export_reference(opposite0, TOWARD_NEGATIVE, SATURATE, &kPolicy) ==
      export_reference(opposite1, TOWARD_NEGATIVE, SATURATE, &kPolicy)) {
    fprintf(stderr, "pair corpus cannot discriminate the lanes\n");
    failed = 1;
  }
  int64_t rail = lane_reference(-32768, -32768, 513);
  if (export_reference(rail, NEAREST_TIES_POSITIVE, SATURATE, &kPolicy) != (1 << 24) ||
      export_reference(rail, TOWARD_NEGATIVE, SATURATE, &kPolicy) != (1 << 24) - 1) {
    fprintf(stderr, "pair corpus lost the rail-carry discrimination\n");
    failed = 1;
  }
  int64_t tie = lane_reference(1, -16384, 3);
  if (export_reference(tie, NEAREST_TIES_POSITIVE, SATURATE, &kPolicy) != -1 ||
      export_reference(tie, TOWARD_NEGATIVE, SATURATE, &kPolicy) != -2 ||
      export_reference(tie, NEAREST_EVEN, SATURATE, &kPolicy) != -2) {
    fprintf(stderr, "pair corpus lost the negative-tie discrimination\n");
    failed = 1;
  }
  return failed;
}

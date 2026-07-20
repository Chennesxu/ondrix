#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define MEMREF1_DECL float *, float *, int64_t, int64_t, int64_t
#define MEMREF2_DECL float *, float *, int64_t, int64_t, int64_t, int64_t, int64_t
#define MEMREF1_ARGS(pointer, size) pointer, pointer, INT64_C(0), size, INT64_C(1)
#define MEMREF2_ARGS(pointer, rows, columns)                                                       \
  pointer, pointer, INT64_C(0), rows, columns, columns, INT64_C(1)
#define CALL_OUTPUT(function, input, length, coefficients, sections, scales, state, index)         \
  function(MEMREF1_ARGS(input, length), MEMREF2_ARGS(coefficients, sections, 5),                   \
           MEMREF1_ARGS(scales, sections), MEMREF2_ARGS(state, sections, 2), index)
#define CALL_STATE(function, input, length, coefficients, sections, scales, state, section, slot)  \
  function(MEMREF1_ARGS(input, length), MEMREF2_ARGS(coefficients, sections, 5),                   \
           MEMREF1_ARGS(scales, sections), MEMREF2_ARGS(state, sections, 2), section, slot)

extern float sos_fma_output_value(MEMREF1_DECL, MEMREF2_DECL, MEMREF1_DECL, MEMREF2_DECL, int64_t);
extern float sos_fma_state_value(MEMREF1_DECL, MEMREF2_DECL, MEMREF1_DECL, MEMREF2_DECL, int64_t,
                                 int64_t);
extern float sos_off_output_value(MEMREF1_DECL, MEMREF2_DECL, MEMREF1_DECL, MEMREF2_DECL, int64_t);
extern float sos_off_state_value(MEMREF1_DECL, MEMREF2_DECL, MEMREF1_DECL, MEMREF2_DECL, int64_t,
                                 int64_t);

typedef float (*Madd)(float, float, float);

static float madd_fma(float lhs, float rhs, float accumulator) {
  return fmaf(lhs, rhs, accumulator);
}

static float madd_off(float lhs, float rhs, float accumulator) {
  float product = lhs * rhs;
  return accumulator + product;
}

static void sos_reference(const float *input, int64_t length, const float *coefficients,
                          int64_t sections, const float *scales, const float *initial_state,
                          float *output, float *next_state, Madd madd) {
  memcpy(next_state, initial_state, (size_t)(sections * 2) * sizeof(float));
  for (int64_t sample = 0; sample < length; ++sample) {
    float value = input[sample];
    for (int64_t section = 0; section < sections; ++section) {
      const float *coefficient = coefficients + section * 5;
      float *state = next_state + section * 2;
      float scaled = value * scales[section];
      float section_output = madd(scaled, coefficient[0], state[0]);
      float feedback1 = section_output * coefficient[3];
      float first_term = madd(scaled, coefficient[1], feedback1);
      float next_z1 = state[1] + first_term;
      float feedback2 = section_output * coefficient[4];
      float next_z2 = madd(scaled, coefficient[2], feedback2);
      state[0] = next_z1;
      state[1] = next_z2;
      value = section_output;
    }
    output[sample] = value;
  }
}

static uint32_t float_bits(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static float invoke_output(float (*function)(MEMREF1_DECL, MEMREF2_DECL, MEMREF1_DECL, MEMREF2_DECL,
                                             int64_t),
                           float *input, int64_t length, float *coefficients, int64_t sections,
                           float *scales, const float *state, int64_t index) {
  float state_copy[4];
  memcpy(state_copy, state, (size_t)(sections * 2) * sizeof(float));
  return CALL_OUTPUT(function, input, length, coefficients, sections, scales, state_copy, index);
}

static float invoke_state(float (*function)(MEMREF1_DECL, MEMREF2_DECL, MEMREF1_DECL, MEMREF2_DECL,
                                            int64_t, int64_t),
                          float *input, int64_t length, float *coefficients, int64_t sections,
                          float *scales, const float *state, int64_t section, int64_t slot) {
  float state_copy[4];
  memcpy(state_copy, state, (size_t)(sections * 2) * sizeof(float));
  return CALL_STATE(function, input, length, coefficients, sections, scales, state_copy, section,
                    slot);
}

static int check_contract(const char *name, Madd madd,
                          float (*output_value)(MEMREF1_DECL, MEMREF2_DECL, MEMREF1_DECL,
                                                MEMREF2_DECL, int64_t),
                          float (*state_value)(MEMREF1_DECL, MEMREF2_DECL, MEMREF1_DECL,
                                               MEMREF2_DECL, int64_t, int64_t)) {
  float input[] = {-0.0f, 0x1.000002p+0f, -0.5f, 0.25f, 2.0f, -1.0f, 0.125f};
  float coefficients[] = {
      0.75f, 0.125f, -0.25f, 0.2f, -0.1f, 1.0f, -0.3f, 0.2f, 0.15f, 0.05f,
  };
  float scales[] = {0.5f, 1.25f};
  float initial_state[] = {0.125f, -0.25f, 0.375f, -0.125f};
  float expected_output[7], expected_state[4];
  sos_reference(input, 7, coefficients, 2, scales, initial_state, expected_output, expected_state,
                madd);

  int failed = 0;
  for (int64_t index = 0; index < 7; ++index) {
    float actual =
        invoke_output(output_value, input, 7, coefficients, 2, scales, initial_state, index);
    if (float_bits(actual) != float_bits(expected_output[index])) {
      fprintf(stderr, "%s whole output %lld: expected 0x%08x, got 0x%08x\n", name, (long long)index,
              float_bits(expected_output[index]), float_bits(actual));
      failed = 1;
    }
  }
  for (int64_t section = 0; section < 2; ++section) {
    for (int64_t slot = 0; slot < 2; ++slot) {
      float actual = invoke_state(state_value, input, 7, coefficients, 2, scales, initial_state,
                                  section, slot);
      float expected = expected_state[section * 2 + slot];
      if (float_bits(actual) != float_bits(expected)) {
        fprintf(stderr, "%s whole state %lld,%lld: expected 0x%08x, got 0x%08x\n", name,
                (long long)section, (long long)slot, float_bits(expected), float_bits(actual));
        failed = 1;
      }
    }
  }

  float split_output[3], split_state[4];
  sos_reference(input, 3, coefficients, 2, scales, initial_state, split_output, split_state, madd);
  for (int64_t section = 0; section < 2; ++section)
    for (int64_t slot = 0; slot < 2; ++slot)
      split_state[section * 2 + slot] = invoke_state(state_value, input, 3, coefficients, 2, scales,
                                                     initial_state, section, slot);
  for (int64_t index = 0; index < 4; ++index) {
    float actual =
        invoke_output(output_value, input + 3, 4, coefficients, 2, scales, split_state, index);
    if (float_bits(actual) != float_bits(expected_output[index + 3])) {
      fprintf(stderr, "%s split output %lld differs\n", name, (long long)index);
      failed = 1;
    }
  }
  for (int64_t section = 0; section < 2; ++section) {
    for (int64_t slot = 0; slot < 2; ++slot) {
      float actual = invoke_state(state_value, input + 3, 4, coefficients, 2, scales, split_state,
                                  section, slot);
      if (float_bits(actual) != float_bits(expected_state[section * 2 + slot])) {
        fprintf(stderr, "%s split state %lld,%lld differs\n", name, (long long)section,
                (long long)slot);
        failed = 1;
      }
    }
  }

  float dummy = 0.0f;
  for (int64_t section = 0; section < 2; ++section) {
    for (int64_t slot = 0; slot < 2; ++slot) {
      float actual = invoke_state(state_value, &dummy, 0, coefficients, 2, scales, initial_state,
                                  section, slot);
      if (float_bits(actual) != float_bits(initial_state[section * 2 + slot])) {
        fprintf(stderr, "%s empty chunk changed state %lld,%lld\n", name, (long long)section,
                (long long)slot);
        failed = 1;
      }
    }
  }
  return failed;
}

int main(void) {
  return check_contract("fma", madd_fma, sos_fma_output_value, sos_fma_state_value) |
         check_contract("off", madd_off, sos_off_output_value, sos_off_state_value);
}

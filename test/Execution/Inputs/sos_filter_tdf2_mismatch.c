#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define MEMREF1_DECL float *, float *, int64_t, int64_t, int64_t
#define MEMREF2_DECL float *, float *, int64_t, int64_t, int64_t, int64_t, int64_t
#define MEMREF1_ARGS(pointer, size) pointer, pointer, INT64_C(0), size, INT64_C(1)
#define MEMREF2_ARGS(pointer, rows, columns)                                                       \
  pointer, pointer, INT64_C(0), rows, columns, columns, INT64_C(1)

extern float sos_off_output_value(MEMREF1_DECL, MEMREF2_DECL, MEMREF1_DECL, MEMREF2_DECL, int64_t);

int main(int argc, char **argv) {
  float input[] = {1.0f};
  float coefficients[10] = {0.0f};
  float scales[] = {1.0f, 1.0f};
  float state[4] = {0.0f};
  int64_t scale_sections = 2;
  int64_t state_sections = 2;
  if (argc != 2)
    return 2;
  if (strcmp(argv[1], "scales") == 0)
    scale_sections = 1;
  else if (strcmp(argv[1], "state") == 0)
    state_sections = 1;
  else
    return 2;

  (void)sos_off_output_value(MEMREF1_ARGS(input, 1), MEMREF2_ARGS(coefficients, 2, 5),
                             MEMREF1_ARGS(scales, scale_sections),
                             MEMREF2_ARGS(state, state_sections, 2), 0);
  fprintf(stderr, "invalid SOS section counts were not rejected\n");
  return 1;
}

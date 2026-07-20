#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define MEMREF_ARGS(pointer, size) pointer, pointer, INT64_C(0), size, INT64_C(1)

extern int16_t q15_stream_output_value(int16_t *, int16_t *, int64_t, int64_t, int64_t, int16_t *,
                                       int16_t *, int64_t, int64_t, int64_t, int16_t *, int16_t *,
                                       int64_t, int64_t, int64_t, int64_t);
extern int16_t q15_stream_static_output_value(int16_t *, int16_t *, int64_t, int64_t, int64_t,
                                              int16_t *, int16_t *, int64_t, int64_t, int64_t,
                                              int16_t *, int16_t *, int64_t, int64_t, int64_t,
                                              int64_t);
extern int16_t q15_stream_static_state_value(int16_t *, int16_t *, int64_t, int64_t, int64_t,
                                             int16_t *, int16_t *, int64_t, int64_t, int64_t,
                                             int16_t *, int16_t *, int64_t, int64_t, int64_t,
                                             int64_t);

int main(int argc, char **argv) {
  int16_t input[] = {1, 2, 3, 4};
  int16_t coefficients[] = {1, 2, 3};
  int16_t state[] = {5, 6};
  int64_t coefficient_length = 3;
  int64_t state_length = 2;
  if (argc != 2)
    return 2;
  if (strcmp(argv[1], "coefficients") == 0)
    coefficient_length = 0;
  else if (strcmp(argv[1], "state") == 0)
    state_length = 1;
  else if (strcmp(argv[1], "output") == 0) {
    (void)q15_stream_static_output_value(MEMREF_ARGS(input, 3), MEMREF_ARGS(coefficients, 3),
                                         MEMREF_ARGS(state, 2), 0);
    fprintf(stderr, "invalid FIR stream output shape was not rejected\n");
    return 1;
  } else if (strcmp(argv[1], "next") == 0) {
    int16_t longer_coefficients[] = {1, 2, 3, 4};
    int16_t longer_state[] = {5, 6, 7};
    (void)q15_stream_static_state_value(MEMREF_ARGS(input, 4), MEMREF_ARGS(longer_coefficients, 4),
                                        MEMREF_ARGS(longer_state, 3), 0);
    fprintf(stderr, "invalid FIR stream next-state shape was not rejected\n");
    return 1;
  } else {
    return 2;
  }
  (void)q15_stream_output_value(MEMREF_ARGS(input, 4),
                                MEMREF_ARGS(coefficients, coefficient_length),
                                MEMREF_ARGS(state, state_length), 0);
  fprintf(stderr, "invalid FIR stream shape was not rejected\n");
  return 1;
}

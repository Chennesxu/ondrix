#include <stdint.h>
#include <string.h>

#define MEMREF_ARGS(pointer, size) pointer, pointer, INT64_C(0), size, INT64_C(1)

extern int16_t q15_fir_filter_value(int16_t *, int16_t *, int64_t, int64_t, int64_t, int16_t *,
                                    int16_t *, int64_t, int64_t, int64_t, int16_t *, int16_t *,
                                    int64_t, int64_t, int64_t, int64_t);

int main(int argc, char **argv) {
  int16_t input[4] = {1, 2, 3, 4};
  int16_t coefficients[5] = {1, 1, 1, 1, 1};
  int16_t output[4] = {0};
  int64_t input_length = 4;
  int64_t coefficient_length = 2;
  int64_t output_length = 3;

  if (argc != 2)
    return 2;
  if (strcmp(argv[1], "empty") == 0)
    coefficient_length = 0;
  else if (strcmp(argv[1], "short") == 0)
    coefficient_length = 5;
  else if (strcmp(argv[1], "output") == 0)
    output_length = 4;
  else
    return 2;

  (void)q15_fir_filter_value(MEMREF_ARGS(input, input_length),
                             MEMREF_ARGS(coefficients, coefficient_length),
                             MEMREF_ARGS(output, output_length), 0);
  return 0;
}

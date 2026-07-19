#include <stdint.h>

extern void q15_full_output_vector(int16_t *, int16_t *, int64_t, int64_t, int64_t, int16_t *,
                                   int16_t *, int64_t, int64_t, int64_t, int16_t *, int16_t *,
                                   int64_t, int64_t, int64_t);
extern void q15_full_boundary_scalar(int16_t *, int16_t *, int64_t, int64_t, int64_t, int16_t *,
                                     int16_t *, int64_t, int64_t, int64_t, int16_t *, int16_t *,
                                     int64_t, int64_t, int64_t);

int main(int argc, char **argv) {
  int16_t input[8] = {0};
  int16_t coefficients[3] = {0};
  int16_t output[9] = {0};
  if (argc != 2)
    return 2;

  int64_t input_length = 8;
  int64_t coefficient_length = 3;
  int64_t output_length = 6;
  int full_boundary = 0;
  switch (argv[1][0]) {
  case 'e':
    coefficient_length = 0;
    output_length = 9;
    break;
  case 's':
    input_length = 2;
    output_length = 0;
    break;
  case 'o':
    output_length = 7;
    break;
  case 'f':
    input_length = 5;
    output_length = 6;
    full_boundary = 1;
    break;
  case 'i':
    input_length = 0;
    output_length = 2;
    full_boundary = 1;
    break;
  case 'c':
    input_length = 5;
    coefficient_length = 0;
    output_length = 4;
    full_boundary = 1;
    break;
  default:
    return 2;
  }

  if (full_boundary) {
    q15_full_boundary_scalar(input, input, 0, input_length, 1, coefficients, coefficients, 0,
                             coefficient_length, 1, output, output, 0, output_length, 1);
  } else {
    q15_full_output_vector(input, input, 0, input_length, 1, coefficients, coefficients, 0,
                           coefficient_length, 1, output, output, 0, output_length, 1);
  }
  return 0;
}

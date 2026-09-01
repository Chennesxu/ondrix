def q31_magnitude(input: tensor[complex_q31,8]) -> tensor[q31,8]:
  return magnitude(cfft(input))

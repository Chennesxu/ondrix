def q31_magnitude_component_even(input: tensor[complex_q31,8]) -> tensor[q31,8]:
  return magnitude(input, input_rounding = nearest_even)

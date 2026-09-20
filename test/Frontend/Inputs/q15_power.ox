def q15_power(input: tensor[complex_q15, 8]) -> tensor[q15, 8]:
  return power(input, rounding=toward_negative)

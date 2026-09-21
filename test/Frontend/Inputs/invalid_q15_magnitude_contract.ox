def invalid_q15_magnitude_contract(input: tensor[complex_q15, 8]) -> tensor[q15, 8]:
  return magnitude(input, contract=off)

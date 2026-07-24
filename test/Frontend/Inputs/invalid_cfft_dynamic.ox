def invalid_cfft_dynamic(input: tensor[complex_q15]) -> tensor[complex_q15]:
  return cfft(input)

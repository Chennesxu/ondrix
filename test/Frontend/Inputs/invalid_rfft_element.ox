def invalid_rfft_element(input: tensor[complex_q15,9]) -> tensor[complex_q15,5]:
  return rfft(input)

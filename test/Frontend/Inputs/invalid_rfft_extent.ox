def invalid_rfft_extent(input: tensor[q15,32]) -> tensor[complex_q15,17]:
  return rfft(input)

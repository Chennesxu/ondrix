def q15_rfft16(input: tensor[q15,16]) -> tensor[complex_q15,9]:
  return rfft(input)

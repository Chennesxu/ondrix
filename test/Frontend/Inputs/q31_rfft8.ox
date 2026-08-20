def q31_rfft8(input: tensor[q31,8]) -> tensor[complex_q31,5]:
  return rfft(input)

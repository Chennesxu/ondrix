def q31_icfft8(input: tensor[complex_q31,8]) -> tensor[complex_q31,8]:
  return icfft(input)

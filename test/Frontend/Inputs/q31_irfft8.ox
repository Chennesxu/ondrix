def q31_irfft8(input: tensor[complex_q31,5]) -> tensor[q31,8]:
  return irfft(input)

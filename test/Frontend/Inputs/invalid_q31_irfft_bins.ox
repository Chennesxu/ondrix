def invalid_q31_irfft_bins(input: tensor[complex_q31,6]) -> tensor[q31,10]:
  return irfft(input)

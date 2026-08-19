def q31_cfft_round_trip(input: tensor[complex_q31,64]) -> tensor[complex_q31,64]:
  return icfft(cfft(input, rounding=toward_negative, overflow=saturate))

def q31_rfft_round_trip(input: tensor[q31,8]) -> tensor[q31,8]:
  return irfft(rfft(input))

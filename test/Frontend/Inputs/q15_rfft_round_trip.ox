def q15_rfft_round_trip(input: tensor[q15,16]) -> tensor[q15,16]:
  return irfft(rfft(input))

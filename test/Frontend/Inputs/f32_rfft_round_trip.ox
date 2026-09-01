def f32_rfft_round_trip(input: tensor[f32,32]) -> tensor[f32,32]:
  return irfft(rfft(input, contract = fma), contract = fma)

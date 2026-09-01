def f32_cfft_round_trip_loops(input: tensor[complex_f32,16]) -> tensor[complex_f32,16]:
  return icfft(cfft(input, contract = off), contract = off)
